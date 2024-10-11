package test

import (
	"fmt"
	"testing"
	"time"

	"github.com/lifejade/cnn_ckksv5/cnn"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"github.com/tuneinsight/lattigo/v5/utils/sampling"
)

// boot params test
func Test_Rotate(t *testing.T) {

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

	conRot := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, -1, -2, -3}
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

	value := make([]complex128, n)
	for i := range value {
		value[i] = complex(float64(i), 0)
	}
	plaintext := hefloat.NewPlaintext(params, 16)
	encoder.Encode(value, plaintext)
	cipher, _ := encryptor.EncryptNew(plaintext)

	cipher2, _ := evaluator.RotateNew(cipher, 1)
	DecryptPrint(params, cipher2, *decryptor, *encoder)
}

func Test_RotateTime(t *testing.T) {

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

	conRot := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, -1, -2, -3}
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

	value := make([]complex128, n)
	for i := range value {
		value[i] = sampling.RandComplex128(-1, 1)
	}
	plaintext := hefloat.NewPlaintext(params, params.MaxLevel())
	encoder.Encode(value, plaintext)
	cipher, _ := encryptor.EncryptNew(plaintext)
	DecryptPrint(params, cipher, *decryptor, *encoder)

	startTime := time.Now()
	cipher_high := cipher.CopyNew()
	for i := 0; i < 30; i++ {
		evaluator.Rotate(cipher_high, 1, cipher_high)
	}
	elapse := time.Since(startTime)
	DecryptPrint(params, cipher_high, *decryptor, *encoder)
	fmt.Printf("high time : %s\n\n\n", elapse)

	startTime = time.Now()
	cipher_low := cipher.CopyNew()
	evaluator.DropLevel(cipher_low, params.MaxLevel())
	for i := 0; i < 30; i++ {
		evaluator.Rotate(cipher_low, 1, cipher_low)
	}
	elapse = time.Since(startTime)
	DecryptPrint(params, cipher_low, *decryptor, *encoder)
	fmt.Printf("low time :%s\n\n\n", elapse)
}

func Test_RotateTimeBootParams(t *testing.T) {

	ckksParams := cnn.CNN_Cifar10_Parameters

	//parameter init
	params, err := hefloat.NewParametersFromLiteral(ckksParams.SchemeParams)
	if err != nil {
		panic(err)
	}
	btpParams, err := bootstrapping.NewParametersFromLiteral(params, ckksParams.BootstrappingParams)
	if err != nil {
		panic(err)
	}

	// generate classes
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)

	fmt.Println("make boot params")
	btpevk, sk2, _ := btpParams.GenEvaluationKeys(sk)
	btp, _ := bootstrapping.NewEvaluator(btpParams, btpevk)

	n := 1 << params.LogMaxSlots()

	conRot := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, -1, -2, -3}
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

	params2 := *btp.GetParameters()
	kgne2 := rlwe.NewKeyGenerator(params2)
	pk2 := kgne2.GenPublicKeyNew(sk2)
	encryptor2 := rlwe.NewEncryptor(params2, pk2)
	decryptor2 := rlwe.NewDecryptor(params2, sk2)
	encoder2 := hefloat.NewEncoder(params2)
	rlk2 := kgne2.GenRelinearizationKeyNew(sk2)
	rtk2 := kgne2.GenGaloisKeysNew(galEls, sk2)
	evk2 := rlwe.NewMemEvaluationKeySet(rlk2, rtk2...)
	evaluator2 := hefloat.NewEvaluator(params2, evk2)
	_, _, _, _ = evaluator2, encoder2, encryptor2, decryptor2

	value := make([]complex128, n)
	for i := range value {
		value[i] = sampling.RandComplex128(-1, 1)
	}
	plaintext := hefloat.NewPlaintext(params2, params2.MaxLevel())
	encoder2.Encode(value, plaintext)
	cipher, _ := encryptor2.EncryptNew(plaintext)
	DecryptPrint(params2, cipher, *decryptor2, *encoder2)

	startTime := time.Now()
	cipher_high := cipher.CopyNew()
	for i := 0; i < 30; i++ {
		evaluator2.Rotate(cipher_high, 1, cipher_high)
	}
	elapse := time.Since(startTime)
	DecryptPrint(params2, cipher_high, *decryptor2, *encoder2)
	fmt.Printf("high time : %s\n\n\n", elapse)

	startTime = time.Now()
	cipher_low := cipher.CopyNew()
	evaluator2.DropLevel(cipher_low, params2.MaxLevel())
	for i := 0; i < 30; i++ {
		evaluator2.Rotate(cipher_low, 1, cipher_low)
	}
	elapse = time.Since(startTime)
	DecryptPrint(params2, cipher_low, *decryptor2, *encoder2)
	fmt.Printf("low time :%s\n\n\n", elapse)
}

func Test_RotateTimeDifferEval(t *testing.T) {

	ckksParams := cnn.CNN_Cifar10_Parameters

	//parameter init
	params, err := hefloat.NewParametersFromLiteral(ckksParams.SchemeParams)
	if err != nil {
		panic(err)
	}
	btpParams, err := bootstrapping.NewParametersFromLiteral(params, ckksParams.BootstrappingParams)
	if err != nil {
		panic(err)
	}

	// generate classes
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)

	fmt.Println("make boot params")
	btpevk, sk2, _ := btpParams.GenEvaluationKeys(sk)
	btp, _ := bootstrapping.NewEvaluator(btpParams, btpevk)

	n := 1 << params.LogMaxSlots()

	conRot := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, -1, -2, -3}
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

	params2 := *btp.GetParameters()
	kgne2 := rlwe.NewKeyGenerator(params2)
	pk2 := kgne2.GenPublicKeyNew(sk2)
	encryptor2 := rlwe.NewEncryptor(params2, pk2)
	decryptor2 := rlwe.NewDecryptor(params2, sk2)
	encoder2 := hefloat.NewEncoder(params2)
	rlk2 := kgne2.GenRelinearizationKeyNew(sk2)
	rtk2 := kgne2.GenGaloisKeysNew(galEls, sk2)
	evk2 := rlwe.NewMemEvaluationKeySet(rlk2, rtk2...)
	evaluator2 := hefloat.NewEvaluator(params2, evk2)
	_, _, _, _ = evaluator2, encoder2, encryptor2, decryptor2

	value := make([]complex128, n)
	for i := range value {
		value[i] = sampling.RandComplex128(-1, 1)
	}
	plaintext := hefloat.NewPlaintext(params2, params2.MaxLevel())
	encoder2.Encode(value, plaintext)
	cipher, _ := encryptor2.EncryptNew(plaintext)
	evaluator2.DropLevel(cipher, params2.MaxLevel())
	DecryptPrint(params2, cipher, *decryptor2, *encoder2)

	startTime := time.Now()
	cipher1 := cipher.CopyNew()
	for i := 0; i < 30; i++ {
		evaluator.Rotate(cipher1, 1, cipher1)
	}
	elapse := time.Since(startTime)
	DecryptPrint(params2, cipher1, *decryptor2, *encoder2)
	fmt.Printf("high time : %s\n\n\n", elapse)

	startTime = time.Now()
	cipher2 := cipher.CopyNew()
	for i := 0; i < 30; i++ {
		evaluator2.Rotate(cipher2, 1, cipher2)
	}
	elapse = time.Since(startTime)
	DecryptPrint(params2, cipher2, *decryptor2, *encoder2)
	fmt.Printf("low time :%s\n\n\n", elapse)
}

func DecryptPrint(params hefloat.Parameters, ciphertext *rlwe.Ciphertext, decryptor rlwe.Decryptor, encoder hefloat.Encoder) {

	N := 1 << params.LogN()
	n := N / 2
	message := make([]complex128, n)
	encoder.Decode(decryptor.DecryptNew(ciphertext), message)

	fmt.Println()
	fmt.Printf("Level: %d (logQ = %d)\n", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))
	fmt.Printf("Scale: 2^%f\n", ciphertext.LogScale())
	fmt.Printf("Values: %6.10f %6.10f %6.10f %6.10f %6.10f...\n", message[0], message[1], message[2], message[3], message[4])

	// max, min
	max, min := 0.0, 1.0
	for _, v := range message {
		if max < real(v) {
			max = real(v)
		}
		if min > real(v) {
			min = real(v)
		}
	}

	fmt.Println("Max, Min value: ", max, " ", min)
	fmt.Println()
}
