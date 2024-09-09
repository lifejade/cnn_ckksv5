package cnn

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"time"

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

func NewTensorCipherFormData(k, h, w, c, t, p, logn, logq int, data []float64, context Context) TensorCipher {
	plaintext := hefloat.NewPlaintext(*context.params_, context.params_.MaxLevel())
	plaintext.Scale = rlwe.NewScale(math.Pow(2.0, float64(logq)))
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
	for i := 0; i < logsize; i++ {

		if int(addSize/int(math.Pow(2, float64(i))))%2 == 1 {
			temp, _ := context.eval_.RotateNew(out, int(addSize/int(math.Pow(2, float64(i+1))))*int(math.Pow(2, float64(i+1)))*gap)
			context.eval_.Add(sum, temp, sum)
		}

		temp, _ := context.eval_.RotateNew(out, int(math.Pow(2, float64(i)))*gap)
		context.eval_.Add(out, temp, out)
	}
	context.eval_.Add(out, sum, out)
	return out
}
func compactGappedConvolutionPrint(cnnIn TensorCipher, co, st, fh, fw int, data, runningVar, constantWeight []float64, epsilon float64, context Context, stage int, output *bufio.Writer) (cnnOut TensorCipher) {

	timeStart := time.Now()
	cnnOut = compactGappedConvolution(cnnIn, co, st, fh, fw, data, runningVar, constantWeight, epsilon, context)
	// cnnOut = compactGappedConvolutionRemoveImaginary(cnnIn, co, st, fh, fw, data, runningVar, constantWeight, epsilon, context)
	elapse := time.Since(timeStart)
	fmt.Printf("time: %s \n", elapse)
	fmt.Print("convolution ", stage, " result\n")
	output.WriteString("time: " + elapse.String() + "\n")
	output.WriteString("convolution " + strconv.Itoa(stage) + " result\n")
	output.Flush()

	// decryptPrint(cnnOut.cipher_, context, 4)
	decryptPrintTxt(cnnOut.cipher_, output, context, 4)
	printParms(cnnOut)

	return cnnOut
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

	//multiplex parallel packing fillter
	get_weight := func(h, w, i, o int) float64 {
		return data[fh*fw*ci*o+fh*fw*i+fw*h+w]
	}
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

	//select one vector
	get_select_one := func(i, i3, i4, i5 int) float64 {
		if ko*ko*i5+ko*(i3%ko)+(i4%ko) == i {
			return constantWeight[i] / math.Sqrt(runningVar[i]+epsilon)
		} else {
			return 0
		}
	}
	selectOne_BNW := make([][]float64, co)
	for i := 0; i < co; i++ {
		selectOne_BNW[i] = make([]float64, n)
		for i3 := 0; i3 < ko*ho; i3++ {
			for i4 := 0; i4 < ko*wo; i4++ {
				for i5 := 0; i5 < to; i5++ {
					selectOne_BNW[i][i4+ko*wo*i3+ko*ko*wo*ho*i5] = get_select_one(i, i3, i4, i5)
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
		ctxtRot[i1] = make([]*rlwe.Ciphertext, fw)
		for i2 := 0; i2 < fw; i2++ {
			ctxtRot[i1][i2] = cnnCtxIn.CopyNew()
			eval.Rotate(ctxtRot[i1][i2], ki*ki*wi*(i1-(fh-1)/2)+ki*(i2-(fw-1)/2), ctxtRot[i1][i2])
		}
	}

	ctZero := ctZero(context)
	ctd := ctZero.CopyNew()
	_, _ = ctZero, initScale

	for i3 := 0; i3 < q; i3++ {
		ctb := ctZero.CopyNew()
		for i1 := 0; i1 < fh; i1++ {
			for i2 := 0; i2 < fw; i2++ {
				temp, _ := eval.MulRelinNew(ctxtRot[i1][i2], weight[i1][i2][i3])
				eval.Add(ctb, temp, ctb)
			}
		}
		eval.RescaleTo(ctb, initScale, ctb)
		ctc := sumSlot(ctb, ki, 1, context)
		ctc = sumSlot(ctc, ki, ki*wi, context)
		ctc = sumSlot(ctc, ti, ki*ki*wi*hi, context)

		for i4 := 0; (i4 < pi) && (i4+pi*i3 < co); i4++ {
			i := pi*i3 + i4
			temp, _ := eval.RotateNew(ctc, -(i/(ko*ko))*ko*ko*ho*wo+(n/pi)*(i%pi)-((i%(ko*ko))/ko)*ko*wo-(i%ko))
			eval.MulRelin(temp, selectOne_BNW[i], temp)
			eval.Add(ctd, temp, ctd)
		}
	}
	eval.RescaleTo(ctd, initScale, ctd)

	result := ctd.CopyNew()
	for j := 0; j < int(log2IntPlusToll(float64(po))); j++ {
		temp, _ := eval.RotateNew(result, -pow2(j)*(n/po))
		eval.Add(result, temp, result)
	}

	cnnOut = TensorCipher{
		k_:      ko,
		h_:      ho,
		w_:      wo,
		c_:      co,
		t_:      to,
		p_:      po,
		logn_:   logn,
		cipher_: result,
	}
	return cnnOut
}

func compactGappedBatchNormPrint(cnnIn TensorCipher, bias, runningMean, runningVar, weight []float64, epsilon, B float64, context Context, stage int, output *bufio.Writer) (cnnOut TensorCipher) {

	timeStart := time.Now()
	cnnOut = compactGappedBatchNorm(cnnIn, bias, runningMean, runningVar, weight, epsilon, B, context)
	elapse := time.Since(timeStart)
	fmt.Printf("time: %s \n", elapse)
	fmt.Print("batch normalization ", stage, " result\n")
	output.WriteString("time: " + elapse.String() + "\n")
	output.WriteString("batch normalization " + strconv.Itoa(stage) + " result\n")
	output.Flush()

	// decryptPrint(cnnOut.cipher_, context, 4)
	decryptPrintTxt(cnnOut.cipher_, output, context, 4)

	printParms(cnnOut)

	return cnnOut
}

func compactGappedBatchNorm(cnnIn TensorCipher, bias, runningMean, runningVar, weight []float64, epsilon, B float64, context Context) (cnnOut TensorCipher) {

	// parameter setting
	ki, hi, wi, ci, ti, pi, logn := cnnIn.k_, cnnIn.h_, cnnIn.w_, cnnIn.c_, cnnIn.t_, cnnIn.p_, cnnIn.logn_
	ko, ho, wo, co, to, po := ki, hi, wi, ci, ti, pi
	// cipherG := context.cipherPool_[1]

	// error check
	if len(bias) != ci || len(runningMean) != ci || len(runningVar) != ci || len(weight) != ci {
		fmt.Println("the size of bias, running_mean, running_var, or weight are not correct")
		os.Exit(1)
	}
	for _, num := range runningVar {
		if num < math.Pow(10, -16) && num > -math.Pow(10, -16) {
			fmt.Println("the size of running_var is too small. nearly zero.")
			os.Exit(1)
		}
	}
	if hi*wi*ci > 1<<logn {
		fmt.Println("hi*wi*ci should not be larger than n")
		os.Exit(1)
	}

	// generate g vector
	g := make([]float64, 1<<logn)

	// set f value
	n := 1 << logn

	// check if pi | n
	if n%pi != 0 {
		fmt.Println("n is not divisible by pi")
		os.Exit(1)
	}

	// set g vector
	for v4 := 0; v4 < n; v4++ {
		v1, v2, u3 := ((v4%(n/pi))%(ki*ki*hi*wi))/(ki*wi), (v4%(n/pi))%(ki*wi), (v4%(n/pi))/(ki*ki*hi*wi)
		if ki*ki*u3+ki*(v1%ki)+v2%ki >= ci || v4%(n/pi) >= ki*ki*hi*wi*ti {
			g[v4] = 0.0
		} else {
			idx := ki*ki*u3 + ki*(v1%ki) + v2%ki
			g[v4] = (runningMean[idx]*weight[idx]/math.Sqrt(runningVar[idx]+epsilon) - bias[idx]) / B
		}
	}
	result, _ := context.eval_.SubNew(cnnIn.cipher_, g)
	cnnOut = TensorCipher{
		logn_:   logn,
		k_:      ko,
		h_:      ho,
		w_:      wo,
		c_:      co,
		t_:      to,
		p_:      po,
		cipher_: result,
	}

	// plain := (*context.encoder_).EncodeNew(g, context.params_.MaxLevel(), ctxtIn.Scale, logn)
	// (*context.encryptor_).Encrypt(plain, cipherG)
	// (*context.eval_).Sub(cnnOut.cipher_, cipherG, cnnOut.cipher_)

	return cnnOut

}

func approxReLUPrint(cnnIn TensorCipher, alpha int, output *bufio.Writer, context Context, stage int) (cnnOut TensorCipher) {

	timeStart := time.Now()
	cnnOut = approxReLU(cnnIn, alpha, context)
	elapse := time.Since(timeStart)
	fmt.Printf("time: %s \n", elapse)
	fmt.Print("ReLU function ", stage, " result\n")
	output.WriteString("time: " + elapse.String() + "\n")
	output.WriteString("ReLU function " + strconv.Itoa(stage) + " result\n")
	output.Flush()

	// decryptPrint(cnnOut.cipher_, context, 4)
	decryptPrintTxt(cnnOut.cipher_, output, context, 4)
	printParms(cnnOut)

	return cnnOut
}
func approxReLU(cnnIn TensorCipher, alpha int, context Context) (cnnOut TensorCipher) {

	// parameter setting
	ki, hi, wi, ci, ti, pi, logn := cnnIn.k_, cnnIn.h_, cnnIn.w_, cnnIn.c_, cnnIn.t_, cnnIn.p_, cnnIn.logn_
	ko, ho, wo, co, to, po := ki, hi, wi, ci, ti, pi
	ctxtIn := cnnIn.cipher_
	var temp *rlwe.Ciphertext

	// error check
	if hi*wi*ci > 1<<logn {
		fmt.Println("hi*wi*ci should not be larger than n")
		os.Exit(1)
	}

	temp = EvalApproxMinimaxReLU(ctxtIn, alpha, context)
	// temp = EvalApproxReLU(ctxtIn, alpha, context.eval_, context.params_)
	// temp = EvalApproxReLUDebug(ctxtIn, alpha, context, context.params_)

	cnnOut = TensorCipher{
		logn_:   logn,
		k_:      ko,
		h_:      ho,
		w_:      wo,
		c_:      co,
		t_:      to,
		p_:      po,
		cipher_: temp,
	}

	return cnnOut

}

func bootstrapImaginaryPrint(cnnIn TensorCipher, context Context, logSlots int, stage int, output *bufio.Writer) (cnnOut TensorCipher) {
	timeStart := time.Now()
	var result *rlwe.Ciphertext

	if logSlots == 14 {
		result, _ = context.btp14_.Bootstrap(cnnIn.cipher_)
	} else if logSlots == 13 {
		result, _ = context.btp13_.Bootstrap(cnnIn.cipher_)
	} else if logSlots == 12 {
		result, _ = context.btp12_.Bootstrap(cnnIn.cipher_)
	} else {
		os.Exit(1)
	}
	temp, _ := context.eval_.ConjugateNew(result)
	context.eval_.Add(result, temp, result)
	elapse := time.Since(timeStart)

	cnnOut = cnnIn
	cnnOut.cipher_ = result

	fmt.Printf("time: %s \n", elapse)
	fmt.Print("bootstrapping ", stage, " result\n")
	output.WriteString("time: " + elapse.String() + "\n")
	output.WriteString("bootstrapping " + strconv.Itoa(stage) + " result\n")
	output.Flush()

	// decryptPrint(cnnOut.cipher_, context, 4)
	decryptPrintTxt(cnnOut.cipher_, output, context, 4)
	printParms(cnnOut)

	return cnnOut
}

func downsamplingPrint(cnnIn TensorCipher, context Context, stage int, output *bufio.Writer) (cnnOut TensorCipher) {

	timeStart := time.Now()
	cnnOut = downsampling(cnnIn, context)
	elapse := time.Since(timeStart)

	fmt.Printf("time: %s \n", elapse)
	fmt.Print("downsampling ", stage, " result\n")
	output.WriteString("time: " + elapse.String() + "\n")
	output.WriteString("downsampling " + strconv.Itoa(stage) + " result\n")
	output.Flush()

	// decryptPrint(cnnOut.cipher_, context, 4)
	decryptPrintTxt(cnnOut.cipher_, output, context, 4)
	printParms(cnnOut)

	return cnnOut

}

func downsampling(cnnIn TensorCipher, context Context) (cnnOut TensorCipher) {

	// parameter setting
	ki, hi, wi, ci, ti, logn := cnnIn.k_, cnnIn.h_, cnnIn.w_, cnnIn.c_, cnnIn.t_, cnnIn.logn_
	n := 1 << logn
	ko, ho, wo, co, to := 2*ki, hi/2, wi/2, 2*ci, ti/2
	po := pow2(floorToInt(math.Log2(float64(n) / float64(ko*ko*ho*wo*to))))
	eval := *context.eval_

	if ti%8 != 0 {
		fmt.Println("ti is not multiple of 8")
		os.Exit(1)
	} else if hi%2 != 0 {
		fmt.Println("hi is not even")
		os.Exit(1)
	} else if wi%2 != 0 {
		fmt.Println("wi is not even")
		os.Exit(1)
	} else if n%po != 0 {
		fmt.Println("n is not divisible by po") // check if po | n
		os.Exit(1)
	}

	// selecting tensor vector setting
	selectOneVec := make([][][]float64, ki)
	for w1 := 0; w1 < ki; w1++ {
		selectOneVec[w1] = make([][]float64, ti)
		for w2 := 0; w2 < ti; w2++ {
			selectOneVec[w1][w2] = make([]float64, 1<<logn)
			for v4 := 0; v4 < 1<<logn; v4++ {
				j5, j6, i7 := (v4%(ki*ki*hi*wi))/(ki*wi), v4%(ki*wi), v4/(ki*ki*hi*wi)
				if v4 < ki*ki*hi*wi*ti && (j5/ki)%2 == 0 && (j6/ki)%2 == 0 && (j5%ki) == w1 && i7 == w2 {
					selectOneVec[w1][w2][v4] = 1.0
				} else {
					selectOneVec[w1][w2][v4] = 0.0
				}
			}
		}
	}

	var ct, sum, temp *rlwe.Ciphertext
	ct = cnnIn.cipher_

	for w1 := 0; w1 < ki; w1++ {
		for w2 := 0; w2 < ti; w2++ {
			temp = ct.CopyNew()
			eval.MulRelin(temp, selectOneVec[w1][w2], temp)

			w3, w4, w5 := ((ki*w2+w1)%(2*ko))/2, (ki*w2+w1)%2, (ki*w2+w1)/(2*ko)
			eval.Rotate(temp, ki*ki*hi*wi*w2+ki*wi*w1-ko*ko*ho*wo*w5-ko*wo*w3-ki*w4-ko*ko*ho*wo*(ti/8), temp)
			// RotatePrintCIFAR10Downsampling(temp, ki*ki*hi*wi*w2+ki*wi*w1-ko*ko*ho*wo*w5-ko*wo*w3-ki*w4-ko*ko*ho*wo*(ti/8), temp, &eval, ki, hi, wi, ci, ti, cnnIn.p_)
			if w1 == 0 && w2 == 0 {
				sum = temp.CopyNew()
			} else {
				eval.Add(sum, temp, sum)
			}
		}
	}
	(*context.eval_).RescaleTo(sum, context.params_.DefaultScale(), sum) // plus rescale
	ct = sum.CopyNew()

	sum = ct.CopyNew()
	for u6 := 1; u6 < po; u6++ {
		temp = ct.CopyNew()
		eval.Rotate(temp, -(n/po)*u6, temp)
		// RotatePrintCIFAR10Downsampling(temp, -(n/po)*u6, temp, &eval, ki, hi, wi, ci, ti, cnnIn.p_)
		eval.Add(sum, temp, sum)
	}
	ct = sum.CopyNew()

	cnnOut = TensorCipher{
		logn_:   logn,
		k_:      ko,
		h_:      ho,
		w_:      wo,
		c_:      co,
		t_:      to,
		p_:      po,
		cipher_: ct,
	}
	return cnnOut

}

func cipherAddPrint(cnnIn1 TensorCipher, cnnIn2 TensorCipher, context Context, stage int, output *bufio.Writer) (cnnOut TensorCipher) {

	timeStart := time.Now()
	k1, h1, w1, c1, t1, p1, logn1 := cnnIn1.k_, cnnIn1.h_, cnnIn1.w_, cnnIn1.c_, cnnIn1.t_, cnnIn1.p_, cnnIn1.logn_
	k2, h2, w2, c2, t2, p2, logn2 := cnnIn2.k_, cnnIn2.h_, cnnIn2.w_, cnnIn2.c_, cnnIn2.t_, cnnIn2.p_, cnnIn2.logn_

	if k1 != k2 || h1 != h2 || w1 != w2 || c1 != c2 || t1 != t2 || p1 != p2 || logn1 != logn2 {
		fmt.Println("the parameters of cnn1 and cnn2 are not the same")
		fmt.Println(k1, h1, w1, c1, t1, p1, logn1)
		fmt.Println(k2, h2, w2, c2, t2, p2, logn2)
		os.Exit(1)
	}

	evaluator := *context.eval_
	temp, _ := evaluator.AddNew(cnnIn1.cipher_, cnnIn2.cipher_)
	cnnOut = cnnIn1
	cnnOut.cipher_ = temp
	elapse := time.Since(timeStart)

	fmt.Printf("time: %s \n", elapse)
	fmt.Print("cipher add ", stage, " result\n")
	output.WriteString("time: " + elapse.String() + "\n")
	output.WriteString("cipher add " + strconv.Itoa(stage) + " result\n")
	output.Flush()

	// decryptPrint(cnnOut.cipher_, context, 4)
	decryptPrintTxt(cnnOut.cipher_, output, context, 4)
	printParms(cnnOut)

	return cnnOut
}

func averagepoolingPrint(cnnIn TensorCipher, B float64, context Context, output *bufio.Writer) (cnnOut TensorCipher) {

	timeStart := time.Now()
	cnnOut = averagepooling(cnnIn, B, context)
	elapse := time.Since(timeStart)

	fmt.Printf("time: %s \n", elapse)
	fmt.Print("average pooling result\n")
	output.WriteString("time: " + elapse.String() + "\n")
	output.WriteString("average pooling result\n")
	output.Flush()

	// decryptPrint(cnnOut.cipher_, context, 4)
	decryptPrintTxt(cnnOut.cipher_, output, context, 4)
	printParms(cnnOut)

	return cnnOut

}

func averagepooling(cnnIn TensorCipher, B float64, context Context) (cnnOut TensorCipher) {

	// parameter setting
	ki, hi, wi, ci, ti, logn := cnnIn.k_, cnnIn.h_, cnnIn.w_, cnnIn.c_, cnnIn.t_, cnnIn.logn_
	ko, ho, wo, co, to := 1, 1, 1, ci, ti
	/*
		if log2Int(hi) == -1 {
			fmt.Println("hi is not power of two")
			os.Exit(1)
		} else if log2Int(wi) == -1 {
			fmt.Println("wi is not power of two")
			os.Exit(1)
		}*/

	var temp, sum *rlwe.Ciphertext
	ct := cnnIn.cipher_
	eval := *context.eval_
	initScale := cnnIn.cipher_.Scale

	// sum_hiwi
	fmt.Println("sum hiwi")
	for x := 0; x < int(log2IntPlusToll(float64(wi))); x++ {
		temp = ct.CopyNew()
		eval.Rotate(temp, pow2(x)*ki, temp)
		// RotatePrintCIFAR10Avgpool(temp, pow2(x)*ki, temp, &eval)
		eval.Add(ct, temp, ct)
	}
	for x := 0; x < int(log2IntPlusToll(float64(wi))); x++ {
		temp = ct.CopyNew()
		eval.Rotate(temp, pow2(x)*ki*ki*wi, temp)
		// RotatePrintCIFAR10Avgpool(temp, pow2(x)*ki*ki*wi, temp, &eval)
		eval.Add(ct, temp, ct)
	}

	// final
	fmt.Println("final")
	selectOne := make([]float64, 1<<logn)
	zero := make([]float64, 1<<logn)
	for s := 0; s < ki; s++ {
		for u := 0; u < ti; u++ {
			p := ki*u + s
			temp = ct.CopyNew()
			eval.Rotate(temp, -p*ki+ki*ki*hi*wi*u+ki*wi*s, temp)
			// RotatePrintCIFAR10Avgpool(temp, -p*ki+ki*ki*hi*wi*u+ki*wi*s, temp, &eval)
			copy(selectOne, zero)
			for i := 0; i < ki; i++ {
				selectOne[(ki*u+s)*ki+i] = B / float64(hi*wi)
			}

			eval.MulRelin(temp, selectOne, temp)
			if u == 0 && s == 0 {
				sum = temp.CopyNew() // double scaling factor
			} else {
				eval.Add(sum, temp, sum)
			}
		}
	}
	eval.RescaleTo(sum, initScale, sum) // if scale >= minScale/2, then rescale

	cnnOut = TensorCipher{
		logn_:   logn,
		k_:      ko,
		h_:      ho,
		w_:      wo,
		c_:      co,
		t_:      to,
		p_:      1,
		cipher_: sum,
	}
	return cnnOut

}

func fullyConnectedPrint(cnnIn TensorCipher, matrix []float64, bias []float64, q, r int, context Context, output *bufio.Writer) (cnnOut TensorCipher) {

	timeStart := time.Now()
	cnnOut = matrixMultiplication(cnnIn, matrix, bias, q, r, context)
	elapse := time.Since(timeStart)

	fmt.Printf("time: %s \n", time.Since(timeStart))
	fmt.Print("fully connected layer result\n")
	output.WriteString("time: " + elapse.String() + "\n")
	output.WriteString("fully connected layer result\n")
	output.Flush()

	// decryptPrint(cnnOut.cipher_, context, 4)
	decryptPrintTxt(cnnOut.cipher_, output, context, 10)
	printParms(cnnOut)

	return cnnOut

}

func matrixMultiplication(cnnIn TensorCipher, matrix []float64, bias []float64, q, r int, context Context) (cnnOut TensorCipher) {

	// parameter setting
	ki, hi, wi, ci, ti, pi, logn := cnnIn.k_, cnnIn.h_, cnnIn.w_, cnnIn.c_, cnnIn.t_, cnnIn.p_, cnnIn.logn_
	ko, ho, wo, co, to, po := ki, hi, wi, ci, ti, pi
	eval := *context.eval_
	initScale := cnnIn.cipher_.Scale

	if len(matrix) != q*r {
		fmt.Println("the size of matrix is not q*r")
		os.Exit(1)
	} else if len(bias) != q {
		fmt.Println("the size of bias is not q")
		os.Exit(1)
	}

	// generate matrix and bias
	W := make([][]float64, q+r-1)
	for i := 0; i < q+r-1; i++ {
		W[i] = make([]float64, 1<<logn)
	}
	for i := 0; i < q; i++ {
		for j := 0; j < r; j++ {
			W[i-j+r-1][i] = matrix[i*r+j]
			if i-j+r-1 < 0 || i-j+r-1 >= q+r-1 {
				fmt.Println("i-j+r-1 is out of range")
				os.Exit(1)
			} else if i*r+j < 0 || i*r+j >= len(matrix) {
				fmt.Println("i*r+j is out of range")
				os.Exit(1)
			}
		}
	}
	b := make([]float64, 1<<logn)
	for z := 0; z < q; z++ {
		b[z] = bias[z]
	}

	// matrix multiplication
	var temp, sum *rlwe.Ciphertext
	ct := cnnIn.cipher_
	for s := 0; s < q+r-1; s++ {
		temp = ct.CopyNew()
		eval.Rotate(temp, r-1-s, temp)
		// RotatePrintCIFAR10FClayer(temp, r-1-s, temp, &eval)
		eval.MulRelin(temp, W[s], temp)
		if s == 0 {
			sum = temp.CopyNew()
		} else {
			eval.Add(sum, temp, sum)
		}
	}
	eval.RescaleTo(sum, initScale, sum)

	cnnOut = TensorCipher{
		logn_:   logn,
		k_:      ko,
		h_:      ho,
		w_:      wo,
		c_:      co,
		t_:      to,
		p_:      po,
		cipher_: sum,
	}
	return cnnOut

}
func decryptPrintTxt(ciphertext *rlwe.Ciphertext, output *bufio.Writer, context Context, num int) {
	params := *context.params_
	decryptor := *context.decryptor_
	encoder := *context.encoder_
	// n := params.Slots()
	valuesTest := make([]complex128, params.LogMaxSlots())
	encoder.Decode(decryptor.DecryptNew(ciphertext), valuesTest)

	fmt.Println()
	fmt.Print("/////////////////////////////////////////////////////////////////////\n")
	fmt.Printf("Level: %d (logQ = %d)\n", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))
	fmt.Printf("Scale: 2^%f\n", ciphertext.LogScale())
	fmt.Printf("ValuesTest: ")
	for i := 0; i < num; i++ {
		fmt.Printf("%6.10f ", valuesTest[i])
		if (i+1)%32 == 0 {
			fmt.Println()
		}
	}

	output.WriteString("\n/////////////////////////////////////////////////////////////////////\n")
	output.WriteString("Level:" + strconv.Itoa(ciphertext.Level()) + " (logQ = " + strconv.Itoa(params.LogQLvl(ciphertext.Level())) + ")\n")
	output.WriteString("Scale: 2^" + fmt.Sprintf("%f", ciphertext.LogScale()) + "\n")
	output.WriteString("ValuesTest: ")
	for i := 0; i < num; i++ {
		output.WriteString(fmt.Sprintf("%6.10f ", valuesTest[i]))
		if (i+1)%32 == 0 {
			output.WriteString("\n")
		}
	}

	fmt.Printf("\n/////////////////////////////////////////////////////////////////////\n\n")
	output.WriteString("\n/////////////////////////////////////////////////////////////////////\n\n")
	output.Flush()
}
func printParms(cnn TensorCipher) {
	fmt.Println("parameters: k:", cnn.k_, ", h:", cnn.h_, ", w:", cnn.w_, ", c:", cnn.c_, ", t:", cnn.t_, ", p:", cnn.p_, ", logn:", cnn.logn_)
	fmt.Println()
}
