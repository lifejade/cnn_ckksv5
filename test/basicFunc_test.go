package test

import (
	"math/big"
	"testing"

	"github.com/lifejade/cnn_ckksv5/cnn"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
	"github.com/tuneinsight/lattigo/v5/utils/sampling"
)

func Test_Conjugate(t *testing.T) {

	ckksparam := hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{51, 46, 46, 46, 46, 46, 46, 51},
		LogP:            []int{51, 51, 51},
		Xs:              ring.Ternary{H: 192},
		LogDefaultScale: 46,
	}

	//parameter init
	params, err := hefloat.NewParametersFromLiteral(ckksparam)
	if err != nil {
		panic(err)
	}

	// generate classes
	kgen := rlwe.NewKeyGenerator(params)

	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)

	n := 1 << params.LogMaxSlots()
	rlk := kgen.GenRelinearizationKeyNew(sk)
	conRot := []int{0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3}
	galEls := make([]uint64, len(conRot))
	for i, x := range conRot {
		galEls[i] = params.GaloisElement(x)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())
	rtk := kgen.GenGaloisKeysNew(galEls, sk)

	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	evaluator := hefloat.NewEvaluator(params, evk)

	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)

	_, _, _, _ = encryptor, decryptor, encoder, evaluator

	value := make([]complex128, n)
	for i := range value {
		value[i] = sampling.RandComplex128(-1, 1)
	}
	plaintext := hefloat.NewPlaintext(params, params.MaxLevel())
	encoder.Encode(value, plaintext)
	cipher, _ := encryptor.EncryptNew(plaintext)

	cipher2, _ := evaluator.ConjugateNew(cipher)

	t.Logf("%d %6.10f, %d %6.10f", cipher.Level(), cipher.LogScale(), cipher2.Level(), cipher2.LogScale())

	result := make([]complex128, n)
	encoder.Decode(decryptor.DecryptNew(cipher2), result)

	t.Logf("%6.10f %6.10f %6.10f %6.10f ... %6.10f %6.10f", value[0], value[1], value[2], value[3], value[n-2], value[n-1])
	t.Logf("%6.10f %6.10f %6.10f %6.10f ... %6.10f %6.10f", result[0], result[1], result[2], result[3], result[n-2], result[n-1])

	//t.Logf("%d %6.10f, %d %6.10f", cipher.Level(), cipher.LogScale(), cipher2.Level(), cipher2.LogScale())

	//evaluator.RescaleTo(cipher2, cipher.Scale, cipher2)

}

func Test_Rescaling(t *testing.T) {

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

	n := 1 << params.LogMaxSlots()
	rlk := kgen.GenRelinearizationKeyNew(sk)

	evk := rlwe.NewMemEvaluationKeySet(rlk)
	evaluator := hefloat.NewEvaluator(params, evk)

	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)

	_, _, _, _ = encryptor, decryptor, encoder, evaluator

	value := make([]complex128, n)
	for i := range value {
		value[i] = complex(float64(i), 0)
	}
	plaintext := hefloat.NewPlaintext(params, 16)
	encoder.Encode(value, plaintext)
	cipher, _ := encryptor.EncryptNew(plaintext)

	cipher2 := cipher.CopyNew()
	constvalue := make([]complex128, n)
	for i := range constvalue {
		constvalue[i] = complex(float64(i+2), 0)
	}
	constplttx := hefloat.NewPlaintext(params, 16)
	scale := new(big.Int)
	scale.Div(params.RingQ().ModulusAtLevel[cipher2.Level()], params.RingQ().ModulusAtLevel[cipher2.Level()-1])
	constplttx.Scale = rlwe.NewScale(scale)
	encoder.Encode(constvalue, constplttx)
	cipher_c, _ := encryptor.EncryptNew(constplttx)

	evaluator.MulRelin(cipher2, cipher_c, cipher2)
	t.Logf("%d %6.10f, %d %6.10f", cipher.Level(), cipher.LogScale(), cipher2.Level(), cipher2.LogScale())
	evaluator.RescaleTo(cipher2, cipher.Scale, cipher2)
	t.Logf("%d %6.10f, %d %6.10f", cipher.Level(), cipher.LogScale(), cipher2.Level(), cipher2.LogScale())

	result := make([]complex128, n)
	encoder.Decode(decryptor.DecryptNew(cipher2), result)

	t.Logf("%6.10f %6.10f %6.10f %6.10f ... %6.10f %6.10f", result[0], result[1], result[2], result[3], result[n-2], result[n-1])

	//t.Logf("%d %6.10f, %d %6.10f", cipher.Level(), cipher.LogScale(), cipher2.Level(), cipher2.LogScale())

	//evaluator.RescaleTo(cipher2, cipher.Scale, cipher2)

}

func Test_ConstMul(t *testing.T) {

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

	n := 1 << params.LogMaxSlots()

	rlk := kgen.GenRelinearizationKeyNew(sk)

	evk := rlwe.NewMemEvaluationKeySet(rlk)
	evaluator := hefloat.NewEvaluator(params, evk)

	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)

	_, _, _, _ = encryptor, decryptor, encoder, evaluator

	value := make([]complex128, n)
	for i := range value {
		value[i] = complex(float64(i), 0)
	}
	plaintext := hefloat.NewPlaintext(params, 16)
	encoder.Encode(value, plaintext)
	cipher, _ := encryptor.EncryptNew(plaintext)

	constvalue := make([]complex128, n)
	for i := range constvalue {
		constvalue[i] = complex(float64(i+2), 0)
	}
	cipher2, _ := evaluator.MulRelinNew(cipher, constvalue)

	constplttx := hefloat.NewPlaintext(params, 16)
	encoder.Encode(constvalue, constplttx)
	cipher3, _ := encryptor.EncryptNew(constplttx)
	evaluator.MulRelin(cipher, cipher3, cipher3)

	evaluator.Rescale(cipher2, cipher2)
	result := make([]complex128, n)
	encoder.Decode(decryptor.DecryptNew(cipher2), result)

	result2 := make([]complex128, n)
	encoder.Decode(decryptor.DecryptNew(cipher3), result2)

	t.Logf("%6.10f %6.10f %6.10f %6.10f ... %6.10f %6.10f", result[0], result[1], result[2], result[3], result[n-2], result[n-1])
	t.Logf("%6.10f %6.10f %6.10f %6.10f ... %6.10f %6.10f", result2[0], result2[1], result2[2], result2[3], result2[n-2], result2[n-1])

	t.Logf("%d %6.10f, %d %6.10f", cipher.Level(), cipher.LogScale(), cipher2.Level(), cipher2.LogScale())
	t.Logf("%d %6.10f, %d %6.10f", cipher.Level(), cipher.LogScale(), cipher3.Level(), cipher3.LogScale())
}

func Test_sub(t *testing.T) {

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

	n := 1 << params.LogMaxSlots()

	rlk := kgen.GenRelinearizationKeyNew(sk)

	evk := rlwe.NewMemEvaluationKeySet(rlk)
	evaluator := hefloat.NewEvaluator(params, evk)

	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)

	_, _, _, _ = encryptor, decryptor, encoder, evaluator

	value := make([]complex128, n)
	for i := range value {
		value[i] = complex(float64(i), 0)
	}
	plaintext := hefloat.NewPlaintext(params, 16)
	encoder.Encode(value, plaintext)
	cipher, _ := encryptor.EncryptNew(plaintext)

	constvalue := make([]complex128, n)
	for i := range constvalue {
		constvalue[i] = complex(float64(i+2), 0)
	}
	cipher2, _ := evaluator.SubNew(cipher, constvalue)

	result := make([]complex128, n)
	encoder.Decode(decryptor.DecryptNew(cipher2), result)

	t.Logf("%6.10f %6.10f %6.10f %6.10f ... %6.10f %6.10f", result[0], result[1], result[2], result[3], result[n-2], result[n-1])

	t.Logf("%d %6.10f, %d %6.10f", cipher.Level(), cipher.LogScale(), cipher2.Level(), cipher2.LogScale())

}
