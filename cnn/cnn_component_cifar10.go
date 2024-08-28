package cnn

import (
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
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
	var plaintext rlwe.Plaintext
	context.encoder_.Encode(data, &plaintext)
	cipher, err := context.encryptor_.EncryptNew(&plaintext)
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

func compactGappedConvolution() {

}
func compactGappedBatchNorm() {

}

func approxReLU() {

}
