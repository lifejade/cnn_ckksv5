package test

import (
	"math"

	"github.com/lifejade/cnn_ckksv5/cnn"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

// boot params test
func test3() {

	ckksParams := cnn.CNN_Cifar10_Parameters

	//parameter init
	params, err := hefloat.NewParametersFromLiteral(ckksParams.SchemeParams)
	if err != nil {
		panic(err)
	}

	// generate classes
	kgen := rlwe.NewKeyGenerator(params)

	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)

	conRot := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 3000}
	galEls := make([]uint64, len(conRot))
	for i, x := range conRot {
		galEls[i] = params.GaloisElement(x)
	}

	rlk := kgen.GenRelinearizationKeyNew(sk)
	rtk := kgen.GenGaloisKeysNew(galEls, sk)

	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	evaluator := hefloat.NewEvaluator(params, evk)

	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)

	_, _, _, _ = encryptor, decryptor, encoder, evaluator

	n := 1 << params.LogMaxSlots()
	value := make([]complex128, n)
	for i := range value {
		if i < 100 {
			value[i] = complex(float64(i), 0)
		} else {
			value[i] = complex(0, 0)
		}
	}
	plaintext := hefloat.NewPlaintext(params, 16)
	encoder.Encode(value, plaintext)
	cipher, _ := encryptor.EncryptNew(plaintext)

	DecryptPrint(params, cipher, *decryptor, *encoder)

	addSize := 5
	gap := 2
	logsize := int(math.Log2(float64(addSize)))

	out := cipher.CopyNew()

	zero := make([]complex128, n)
	plaintext2 := hefloat.NewPlaintext(params, 16)
	encoder.Encode(zero, plaintext2)
	sum, _ := encryptor.EncryptNew(plaintext2)
	for i := 0; i < logsize; i++ {
		if int(addSize/int(math.Pow(2, float64(i))))%2 == 1 {
			temp, _ := evaluator.RotateNew(out, int(addSize/int(math.Pow(2, float64(i+1))))*int(math.Pow(2, float64(i+1)))*gap)
			evaluator.Add(sum, temp, sum)
		}

		temp, _ := evaluator.RotateNew(out, int(math.Pow(2, float64(i)))*gap)
		evaluator.Add(out, temp, out)
	}
	evaluator.Add(out, sum, out)
	DecryptPrint(params, out, *decryptor, *encoder)
}

func sumSlot(in *rlwe.Ciphertext, addSize, gap int, eval *hefloat.Evaluator, enc *rlwe.Encryptor) (out *rlwe.Ciphertext) {
	logsize := int(math.Log2(float64(addSize)))

	out = in.CopyNew()
	sum := enc.EncryptZeroNew(out.Level())
	for i := 0; i < logsize; i++ {
		if int(addSize/int(math.Pow(2, float64(i))))%2 == 1 {
			temp, _ := eval.RotateNew(out, int(addSize/int(math.Pow(2, float64(i+1))))*int(math.Pow(2, float64(i+1)))*gap)
			eval.Add(sum, temp, sum)
		}

		temp, _ := eval.RotateNew(out, int(math.Pow(2, float64(i)))*gap)
		eval.Add(out, temp, out)
	}
	eval.Add(out, sum, out)
	return out
}
