package cnn

import (
	"fmt"
	"os"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"

	"math"
)

type TensorCipher struct {
	k_      int
	h_      int
	w_      int
	c_      int
	t_      int
	p_      int
	logn_   int
	cipher_ *rlwe.Ciphertext
}

func NewTensorCipherFormData(k, h, w, c, t, p, logn int, data []float64, context Context) TensorCipher {
	plaintext := hefloat.NewPlaintext(*context.params_, context.params_.MaxLevel())
	context.encoder_.Encode(data, plaintext)
	cipher, err := context.encryptor_.EncryptNew(plaintext)
	if err != nil {
		panic(err)
	}

	return NewTensorCipher(k, h, w, c, t, p, logn, cipher)
}

func NewTensorCipher(k, h, w, c, t, p, logn int, cipher *rlwe.Ciphertext) TensorCipher {
	result := TensorCipher{
		k_:      k,
		h_:      h,
		w_:      w,
		c_:      c,
		t_:      t,
		p_:      p,
		logn_:   logn,
		cipher_: cipher,
	}
	return result
}

type Context struct {
	encoder_   *hefloat.Encoder
	encryptor_ *rlwe.Encryptor
	decryptor_ *rlwe.Decryptor
	sk_        *rlwe.SecretKey
	pk_        *rlwe.PublicKey
	btp14_     *bootstrapping.Evaluator
	btp13_     *bootstrapping.Evaluator
	btp12_     *bootstrapping.Evaluator
	rotkeys_   []*rlwe.GaloisKey
	rlk_       *rlwe.RelinearizationKey
	eval_      *hefloat.Evaluator
	params_    *hefloat.Parameters
}

func NewContext(encoder *hefloat.Encoder, encryptor *rlwe.Encryptor, decryptor *rlwe.Decryptor, sk *rlwe.SecretKey,
	pk *rlwe.PublicKey, btp14 *bootstrapping.Evaluator, btp13 *bootstrapping.Evaluator, btp12 *bootstrapping.Evaluator, rotkeys []*rlwe.GaloisKey, rlk *rlwe.RelinearizationKey,
	eval *hefloat.Evaluator, params *hefloat.Parameters) Context {
	result := Context{
		encoder_:   encoder,
		encryptor_: encryptor,
		decryptor_: decryptor,
		sk_:        sk,
		pk_:        pk,
		btp14_:     btp14,
		btp13_:     btp13,
		btp12_:     btp12,
		rotkeys_:   rotkeys,
		rlk_:       rlk,
		eval_:      eval,
		params_:    params,
	}
	return result
}

func sumSlot(in *rlwe.Ciphertext, addSize, gap int, context Context) (out *rlwe.Ciphertext) {
	out = in.CopyNew()
	sum := ctZero(context)

	logsize := int(math.Log2(float64(addSize)))
	fmt.Println(logsize)
	for i := 0; i < logsize; i++ {

		if int(addSize/int(math.Pow(2, float64(i))))%2 == 1 {
			temp, _ := context.eval_.RotateNew(out, int(addSize/int(math.Pow(2, float64(i+1))))*int(math.Pow(2, float64(i+1)))*gap)
			fmt.Println(int(addSize / int(math.Pow(2, float64(i)))))
			context.eval_.Add(sum, temp, sum)
		}

		temp, _ := context.eval_.RotateNew(out, int(math.Pow(2, float64(i)))*gap)
		context.eval_.Add(out, temp, out)
	}
	context.eval_.Add(out, sum, out)
	return out
}

func compactGappedConvolution(cnnIn TensorCipher, co, st, fh, fw int, data, runningVar, constantWeight []float64, epsilon float64, context Context) (cnnOut TensorCipher) {
	// set parameters
	ki, hi, wi, ci, ti, pi, logn := cnnIn.k_, cnnIn.h_, cnnIn.w_, cnnIn.c_, cnnIn.t_, cnnIn.p_, cnnIn.logn_
	var ko, ho, wo, to, po int

	// evaluator
	eval := *context.eval_

	_, _, _ = po, ti, eval
	// error check
	if st != 1 && st != 2 {
		fmt.Println("supported st is only 1 or 2")
		os.Exit(1)
	}
	if len(data) != fh*fw*ci*co {
		fmt.Println("the size of data vector is not ker x ker x h x h")
		os.Exit(1)
	}
	if len(runningVar) != co || len(constantWeight) != co {
		fmt.Println("the size of running_var or weight is not correct")
		os.Exit(1)
	}

	if st == 1 {
		ho, wo, ko = hi, wi, ki
	} else if st == 2 {
		ho, wo, ko = hi/2, wi/2, ki*2
	}

	n := 1 << logn
	to = ceilToInt(float64(co) / float64(ko*ko))
	po = pow2(ceilToInt(log2IntPlusToll(float64(n) / float64(ko*ko*ho*wo*to))))
	q := ceilToInt(float64(co) / float64(pi))

	// check if pi, po | n
	if n%pi != 0 {
		fmt.Println("n is not divisible by pi")
		os.Exit(1)
	}
	if n%po != 0 {
		fmt.Println("n is not divisible by po")
		os.Exit(1)
	}

	get_weight := func(h, w, i, o int) float64 {
		return data[fh*fw*ci*o+fh*fw*i+fw*h+w]
	}

	//multiplex parallel packing fillter
	weight := make([][][][]float64, fh)
	for i1 := 0; i1 < fh; i1++ {
		weight[i1] = make([][][]float64, fw)
		for i2 := 0; i2 < fw; i2++ {
			weight[i1][i2] = make([][]float64, q)
			for i3 := 0; i3 < q; i3++ {
				weight[i1][i2][i3] = make([]float64, n)
				for i := 0; i < n; i++ {
					temp := i
					i6 := temp % (ki * wi)
					temp /= (ki * wi)
					i5 := temp % (ki * hi)
					temp /= (ki * hi)
					i7 := temp
					var pow2fit_ti int = n / (ki * ki * wi * hi * pi)
					if (i7%pow2fit_ti) >= ti || ki*ki*(i7%pow2fit_ti)+ki*(i5%ki)+(i6%ki) >= ci || (i7/pow2fit_ti)+pi*i3 >= co ||
						(i5/ki)-(fh-1)/2+i1 < 0 || (i5/ki)-(fh-1)/2+i1 > hi-1 || (i6/ki)-(fw-1)/2+i2 < 0 || (i6/ki)-(fw-1)/2+i2 > wi-1 {
						weight[i1][i2][i3][i] = 0
					} else {
						weight[i1][i2][i3][i] = get_weight(i1, i2, ki*ki*(i7%pow2fit_ti)+ki*(i5%ki)+(i6%ki), pi*i3+(i7/pow2fit_ti))
					}
				}
			}
		}
	}

	cnnCtxIn := cnnIn.cipher_
	initScale := cnnCtxIn.Scale

	if fh%2 == 0 || fw%2 == 0 {
		fmt.Println("fh and fw should be odd")
		os.Exit(1)
	}

	// rotated input precomputation
	ctxtRot := make([][]*rlwe.Ciphertext, fh)
	for i1 := 0; i1 < fh; i1++ {
		for i2 := 0; i2 < fw; i2++ {
			ctxtRot[i1] = make([]*rlwe.Ciphertext, fw)
			ctxtRot[i1][i2] = cnnCtxIn.CopyNew()
			eval.Rotate(ctxtRot[i1][i2], (ki*ki*wi*(i1-(fh-1)/2)+ki*(i2-(fw-1)/2)+10*n)%n, ctxtRot[i1][i2])
		}
	}

	ctZero := ctZero(context)
	_, _ = ctZero, initScale

	cnnOut = cnnIn
	return cnnOut
}

func compactGappedBatchNorm() {

}

func approxReLU() {

}
