package test

import (
	"fmt"

	"github.com/lifejade/cnn_ckksv5/cnn"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
)

// boot params test
func Test1() {

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
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)

	fmt.Println("make boot params")
	btpevk, _, _ := btpParams.GenEvaluationKeys(sk)
	btp, _ := bootstrapping.NewEvaluator(btpParams, btpevk)
	_, _, _, _, _ = encryptor, decryptor, encoder, evaluator, btp

	fmt.Println("end boot params")

	n := 1 << params.LogMaxSlots()
	value := make([]complex128, n)
	for i := range value {
		//value[i] = sampling.RandComplex128(-1, 1)
		value[i] = complex(0.05, 0)
		//value[i] = complex(float64(i%30), 0)
	}

	plaintext := hefloat.NewPlaintext(params, 16)
	encoder.Encode(value, plaintext)
	cipher, _ := encryptor.EncryptNew(plaintext)

	DecryptPrint(params, cipher, *decryptor, *encoder)

	res, _ := btp.Bootstrap(cipher)

	DecryptPrint(params, res, *decryptor, *encoder)
}
