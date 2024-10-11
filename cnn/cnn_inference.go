package cnn

import (
	"fmt"
	"math/big"
	"runtime"
	"sync"
	"time"

	"bufio"
	"log"
	"os"
	"strconv"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
)

/*
func ResNetCifar10Multiple(layerNum, startImageId, endImageId int) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	numThreads := endImageId - startImageId + 1
	//check layernumber
	if !(layerNum == 20 || layerNum == 32 || layerNum == 44 || layerNum == 56 || layerNum == 110) {
		fmt.Println("layer_num is not correct")
		os.Exit(1)
	}

	//ckks parameter init
	ckksParams := CNN_Cifar10_Parameters
	initparams, err := hefloat.NewParametersFromLiteral(ckksParams.SchemeParams)
	if err != nil {
		panic(err)
	}
	fmt.Println("ckks parameter init end")

	//boot parameter init
	//Todo : not RS-Bootstrapping yet

	bootParams14, bootParams13, bootParams12 := ckksParams.BootstrappingParams, ckksParams.BootstrappingParams, ckksParams.BootstrappingParams
	//bootParams14.LogSlots, bootParams13.LogSlots, bootParams12.LogSlots = utils.Pointy(14), utils.Pointy(13), utils.Pointy(12)
	btpParams14, err := bootstrapping.NewParametersFromLiteral(initparams, bootParams14)
	if err != nil {
		panic(err)
	}
	btpParams13, err := bootstrapping.NewParametersFromLiteral(initparams, bootParams13)
	if err != nil {
		panic(err)
	}
	btpParams12, err := bootstrapping.NewParametersFromLiteral(initparams, bootParams12)
	if err != nil {
		panic(err)
	}
	btpParams12.SlotsToCoeffsParameters.Scaling, btpParams13.SlotsToCoeffsParameters.Scaling,
		btpParams14.SlotsToCoeffsParameters.Scaling = new(big.Float).SetFloat64(0.5), new(big.Float).SetFloat64(0.5), new(big.Float).SetFloat64(0.5)

	//bootParams14 := ckksParams.BootstrappingParams
	//btpParams14, err := bootstrapping.NewParametersFromLiteral(params,bootParams14)
	fmt.Println("bootstrapping parameter init end")

	initkgen := rlwe.NewKeyGenerator(initparams)
	initsk := initkgen.GenSecretKeyNew()

	//generate bootstrapper
	var wg sync.WaitGroup
	wg.Add(3)
	btp14 := make([]*bootstrapping.Evaluator,numThreads)
	btp13 := make([]*bootstrapping.Evaluator,numThreads)
	btp12 := make([]*bootstrapping.Evaluator,numThreads)
	var sk *rlwe.SecretKey
	for i := 0; i < numThreads; i++ {
		go func() {
			defer wg.Done()
			btpevk14, sk_, _ := btpParams14.GenEvaluationKeys(initsk)
			btp14[i], _ = bootstrapping.NewEvaluator(btpParams14, btpevk14)
			sk = sk_
		}()
		go func() {
			defer wg.Done()
			btpevk13, _, _ := btpParams13.GenEvaluationKeys(initsk)
			btp13[i], _ = bootstrapping.NewEvaluator(btpParams13, btpevk13)
		}()
		go func() {
			defer wg.Done()
			btpevk12, _, _ := btpParams12.GenEvaluationKeys(initsk)
			btp12[i], _ = bootstrapping.NewEvaluator(btpParams12, btpevk12)
		}()
		wg.Wait()
	}

	fmt.Println("generated bootstrapper end")

	params := *btp14[0].GetParameters()
	kgen := rlwe.NewKeyGenerator(params)
	pk := kgen.GenPublicKeyNew(sk)

	// generate keys - Rotating key
	convRot := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
		34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
		56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 84, 124, 128, 132, 256, 512, 959, 960, 990, 991, 1008,
		1023, 1024, 1036, 1064, 1092, 1952, 1982, 1983, 2016, 2044, 2047, 2048, 2072, 2078, 2100, 3007, 3024, 3040, 3052, 3070, 3071, 3072, 3080, 3108, 4031,
		4032, 4062, 4063, 4095, 4096, 5023, 5024, 5054, 5055, 5087, 5118, 5119, 5120, 6047, 6078, 6079, 6111, 6112, 6142, 6143, 6144, 7071, 7102, 7103, 7135,
		7166, 7167, 7168, 8095, 8126, 8127, 8159, 8190, 8191, 8192, 9149, 9183, 9184, 9213, 9215, 9216, 10173, 10207, 10208, 10237, 10239, 10240, 11197, 11231,
		11232, 11261, 11263, 11264, 12221, 12255, 12256, 12285, 12287, 12288, 13214, 13216, 13246, 13278, 13279, 13280, 13310, 13311, 13312, 14238, 14240,
		14270, 14302, 14303, 14304, 14334, 14335, 15262, 15264, 15294, 15326, 15327, 15328, 15358, 15359, 15360, 16286, 16288, 16318, 16350, 16351, 16352,
		16382, 16383, 16384, 17311, 17375, 18335, 18399, 18432, 19359, 19423, 20383, 20447, 20480, 21405, 21406, 21437, 21469, 21470, 21471, 21501, 21504,
		22429, 22430, 22461, 22493, 22494, 22495, 22525, 22528, 23453, 23454, 23485, 23517, 23518, 23519, 23549, 24477, 24478, 24509, 24541, 24542, 24543,
		24573, 24576, 25501, 25565, 25568, 25600, 26525, 26589, 26592, 26624, 27549, 27613, 27616, 27648, 28573, 28637, 28640, 28672, 29600, 29632, 29664,
		29696, 30624, 30656, 30688, 30720, 31648, 31680, 31712, 31743, 31744, 31774, 32636, 32640, 32644, 32672, 32702, 32704, 32706, 32735,
		32736, 32737, 32759, 32760, 32761, 32762, 32763, 32764, 32765, 32766, 32767}
	galEls := make([]uint64, len(convRot))
	for i, x := range convRot {
		galEls[i] = params.GaloisElement(x)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())
	rlk := kgen.GenRelinearizationKeyNew(sk)
	var rtk []*rlwe.GaloisKey = make([]*rlwe.GaloisKey, len(galEls))
	wg.Add(len(galEls))
	for i := range galEls {
		i := i

		go func() {
			defer wg.Done()
			kgen_ := rlwe.NewKeyGenerator(params)
			rtk[i] = kgen_.GenGaloisKeyNew(galEls[i], sk)
		}()
	}
	wg.Wait()
	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)

	//generate -or/er
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)
	fmt.Println("generate Evaluator end")

	context := NewContext(encoder, encryptor, decryptor, sk, pk, btp14, btp13, btp12, rtk, rlk, evaluator, &params)
	// load inference image
	var file *os.File
	file, err := os.Open("testFiles/test_values.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	in := bufio.NewScanner(file)
	SkipLines(in, 32*32*3*imageID)
	var image []float64
	image = ReadLines(in, image, 32*32*3)

	//result file init
	var result_file *os.File
	result_file, _ = os.Create("result/resnet" + strconv.Itoa(layerNum) + "_cifar10_label" + strconv.Itoa(imageID) + ".txt")
	outShare := bufio.NewWriter(result_file)
	_ = outShare
	fmt.Println("Resnet-CKKS result name: ", result_file.Name())
}
*/

func ResNetCifar10MultipleImage(layerNum int, startImageID, endImageID int) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//check layernumber
	if !(layerNum == 20 || layerNum == 32 || layerNum == 44 || layerNum == 56 || layerNum == 110) {
		fmt.Println("layer_num is not correct")
		os.Exit(1)
	}
	// load inference image
	var file *os.File
	file, err := os.Open("testFiles/test_values.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	in := bufio.NewScanner(file)
	SkipLines(in, 32*32*3*startImageID)

	threadNum := endImageID - startImageID + 1
	var images [][]float64
	images = make([][]float64, threadNum)
	for i := 0; i < threadNum; i++ {
		images[i] = ReadLines(in, images[i], 32*32*3)
	}

	//ckks parameter init
	ckksParams := CNN_Cifar10_Parameters
	initparams, err := hefloat.NewParametersFromLiteral(ckksParams.SchemeParams)
	if err != nil {
		panic(err)
	}
	fmt.Println("ckks parameter init end")

	//boot parameter init
	//Todo : not RS-Bootstrapping yet

	bootParams14, bootParams13, bootParams12 := ckksParams.BootstrappingParams, ckksParams.BootstrappingParams, ckksParams.BootstrappingParams
	//bootParams14.LogSlots, bootParams13.LogSlots, bootParams12.LogSlots = utils.Pointy(14), utils.Pointy(13), utils.Pointy(12)
	btpParams14, err := bootstrapping.NewParametersFromLiteral(initparams, bootParams14)
	if err != nil {
		panic(err)
	}
	btpParams13, err := bootstrapping.NewParametersFromLiteral(initparams, bootParams13)
	if err != nil {
		panic(err)
	}
	btpParams12, err := bootstrapping.NewParametersFromLiteral(initparams, bootParams12)
	if err != nil {
		panic(err)
	}
	btpParams12.SlotsToCoeffsParameters.Scaling, btpParams13.SlotsToCoeffsParameters.Scaling,
		btpParams14.SlotsToCoeffsParameters.Scaling = new(big.Float).SetFloat64(0.5), new(big.Float).SetFloat64(0.5), new(big.Float).SetFloat64(0.5)

	//bootParams14 := ckksParams.BootstrappingParams
	//btpParams14, err := bootstrapping.NewParametersFromLiteral(params,bootParams14)
	fmt.Println("bootstrapping parameter init end")

	// generate keys
	//fmt.Println("generate keys")
	//keytime := time.Now()
	initkgen := rlwe.NewKeyGenerator(initparams)
	initsk := initkgen.GenSecretKeyNew()
	contexts := make([]Context, threadNum)

	var params hefloat.Parameters
	var pk *rlwe.PublicKey
	var sk *rlwe.SecretKey
	var btpevk14, btpevk13, btpevk12 *bootstrapping.EvaluationKeys
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	for i := 0; i < threadNum; i++ {
		//generate bootstrapper
		if i == 0 {
			btpevk14, sk, _ = btpParams14.GenEvaluationKeys(initsk)
			btpevk13, _, _ = btpParams13.GenEvaluationKeys(initsk)
			btpevk12, _, _ = btpParams12.GenEvaluationKeys(initsk)
		}

		var btp14, btp13, btp12 *bootstrapping.Evaluator
		btp14, _ = bootstrapping.NewEvaluator(btpParams14, btpevk14)
		btp13, _ = bootstrapping.NewEvaluator(btpParams13, btpevk13)
		btp12, _ = bootstrapping.NewEvaluator(btpParams12, btpevk12)

		fmt.Println("generated bootstrapper end")
		if i == 0 {
			params = *btp14.GetParameters()
		}
		kgen := rlwe.NewKeyGenerator(params)
		if i == 0 {
			pk = kgen.GenPublicKeyNew(sk)
			rlk = kgen.GenRelinearizationKeyNew(sk)
			// generate keys - Rotating key
			convRot := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
				34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
				56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 84, 124, 128, 132, 256, 512, 959, 960, 990, 991, 1008,
				1023, 1024, 1036, 1064, 1092, 1952, 1982, 1983, 2016, 2044, 2047, 2048, 2072, 2078, 2100, 3007, 3024, 3040, 3052, 3070, 3071, 3072, 3080, 3108, 4031,
				4032, 4062, 4063, 4095, 4096, 5023, 5024, 5054, 5055, 5087, 5118, 5119, 5120, 6047, 6078, 6079, 6111, 6112, 6142, 6143, 6144, 7071, 7102, 7103, 7135,
				7166, 7167, 7168, 8095, 8126, 8127, 8159, 8190, 8191, 8192, 9149, 9183, 9184, 9213, 9215, 9216, 10173, 10207, 10208, 10237, 10239, 10240, 11197, 11231,
				11232, 11261, 11263, 11264, 12221, 12255, 12256, 12285, 12287, 12288, 13214, 13216, 13246, 13278, 13279, 13280, 13310, 13311, 13312, 14238, 14240,
				14270, 14302, 14303, 14304, 14334, 14335, 15262, 15264, 15294, 15326, 15327, 15328, 15358, 15359, 15360, 16286, 16288, 16318, 16350, 16351, 16352,
				16382, 16383, 16384, 17311, 17375, 18335, 18399, 18432, 19359, 19423, 20383, 20447, 20480, 21405, 21406, 21437, 21469, 21470, 21471, 21501, 21504,
				22429, 22430, 22461, 22493, 22494, 22495, 22525, 22528, 23453, 23454, 23485, 23517, 23518, 23519, 23549, 24477, 24478, 24509, 24541, 24542, 24543,
				24573, 24576, 25501, 25565, 25568, 25600, 26525, 26589, 26592, 26624, 27549, 27613, 27616, 27648, 28573, 28637, 28640, 28672, 29600, 29632, 29664,
				29696, 30624, 30656, 30688, 30720, 31648, 31680, 31712, 31743, 31744, 31774, 32636, 32640, 32644, 32672, 32702, 32704, 32706, 32735,
				32736, 32737, 32759, 32760, 32761, 32762, 32763, 32764, 32765, 32766, 32767}
			galEls := make([]uint64, len(convRot))
			for i, x := range convRot {
				galEls[i] = params.GaloisElement(x)
			}
			galEls = append(galEls, params.GaloisElementForComplexConjugation())

			rtk = make([]*rlwe.GaloisKey, len(galEls))
			var wg sync.WaitGroup
			wg.Add(len(galEls))
			for i := range galEls {
				i := i

				go func() {
					defer wg.Done()
					kgen_ := rlwe.NewKeyGenerator(params)
					rtk[i] = kgen_.GenGaloisKeyNew(galEls[i], sk)
				}()
			}
			wg.Wait()
		}

		evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
		//generate -er
		encryptor := rlwe.NewEncryptor(params, pk)
		decryptor := rlwe.NewDecryptor(params, sk)
		encoder := hefloat.NewEncoder(params)
		evaluator := hefloat.NewEvaluator(params, evk)
		fmt.Println("generate Evaluator end")

		contexts[i] = NewContext(encoder, encryptor, decryptor, sk, pk, btp14, btp13, btp12, rtk, rlk, evaluator, &params)
	}

	// result image label
	allTimeStart := time.Now()
	var wg sync.WaitGroup
	wg.Add(threadNum)
	for i := 0; i < threadNum; i++ {
		image := images[i]
		imageID := startImageID + i
		context := contexts[i]
		go func() {
			defer wg.Done()
			var buffer []int
			var file *os.File
			file, err = os.Open("testFiles/test_label.txt")
			if err != nil {
				log.Fatal(err)
			}
			in = bufio.NewScanner(file)
			SkipLines(in, imageID)
			buffer = ReadLinesInt(in, buffer, 1)
			imageLabel := buffer[0]
			// output files
			file, _ = os.Create("log/resnet" + strconv.Itoa(layerNum) + "_cifar10_image" + strconv.Itoa(imageID) + ".txt")
			log_writer := bufio.NewWriter(file)
			//result file init
			file, _ = os.Create("result/resnet" + strconv.Itoa(layerNum) + "_cifar10_label" + strconv.Itoa(imageID) + ".txt")
			result_writer := bufio.NewWriter(file)
			defer file.Close()

			label, maxScore := ResNetCifar10(layerNum, image, context, log_writer)
			fmt.Println("image label: ", imageLabel)
			fmt.Println("inferred label: ", label)
			fmt.Println("max score: ", maxScore)
			log_writer.WriteString("image label: " + strconv.Itoa(imageLabel) + "\n")
			log_writer.WriteString("inferred label: " + strconv.Itoa(label) + "\n")
			log_writer.WriteString("max score: " + fmt.Sprintf("%f", maxScore) + "\n")
			log_writer.Flush()
			result_writer.WriteString("image_id: " + strconv.Itoa(imageID) + ", image label: " + strconv.Itoa(imageLabel) + ", inferred label: " + strconv.Itoa(label) + "\n")
			result_writer.Flush()
		}()
	}
	wg.Wait()
	allElapse := time.Since(allTimeStart)
	fmt.Printf("all threads time : %s \n", allElapse)
}

func ResNetCifar10(layerNum int, image []float64, context Context, log_writer *bufio.Writer) (inferredLabel int, maxScore float64) {
	evaluator := context.eval_
	params := context.params_
	encoder, decryptor := context.encoder_, context.decryptor_
	co, st, fh, fw := 0, 0, 3, 3
	init_p := 8
	logp := 46
	logq := 51
	logn := 15
	n := 1 << logn
	stage := 0
	epsilon := 0.00001
	B := 40.0
	linWgt, linBias, convWgt, bnBias, bnMean, bnVar, bnWgt, endNum := ImportParametersCifar10(layerNum)

	for len(image) < n {
		image = append(image, 0.0)
	}
	for i := n / init_p; i < n; i++ {
		image[i] = image[i%(n/init_p)]
	}
	for i := 0; i < n; i++ {
		image[i] /= B
	}

	cnn := NewTensorCipherFormData(1, 32, 32, 3, 3, init_p, logn, logq, image, context)
	fmt.Println("preprocess & encrypt inference image end")

	fmt.Println("drop level")
	evaluator.DropLevel(cnn.cipher_, 11)

	totalTimeStart := time.Now()

	cnn = compactGappedConvolutionPrint(cnn, 16, 1, fh, fw, convWgt[stage], bnVar[stage], bnWgt[stage], epsilon, context, stage, log_writer)
	cnn = compactGappedBatchNormPrint(cnn, bnBias[stage], bnMean[stage], bnVar[stage], bnWgt[stage], epsilon, B, context, stage, log_writer)
	evaluator.SetScale(cnn.cipher_, rlwe.NewScale(1<<logp))
	cnn = approxReLUPrint(cnn, 13, log_writer, context, stage)

	for largeBlockID := 0; largeBlockID < 3; largeBlockID++ {
		if largeBlockID == 0 {
			co = 16
		} else if largeBlockID == 1 {
			co = 32
		} else if largeBlockID == 2 {
			co = 64
		}

		for blockID := 0; blockID <= endNum; blockID++ {
			stage = 2*((endNum+1)*largeBlockID+blockID) + 1
			fmt.Println("layer ", stage)
			tempCnn := TensorCipher{k_: cnn.k_, h_: cnn.h_, w_: cnn.w_, c_: cnn.c_, t_: cnn.t_, p_: cnn.p_, logn_: cnn.logn_, cipher_: cnn.cipher_.CopyNew()}
			if largeBlockID >= 1 && blockID == 0 {
				st = 2
			} else {
				st = 1
			}
			cnn = compactGappedConvolutionPrint(cnn, co, st, fh, fw, convWgt[stage], bnVar[stage], bnWgt[stage], epsilon, context, stage, log_writer)
			cnn = compactGappedBatchNormPrint(cnn, bnBias[stage], bnMean[stage], bnVar[stage], bnWgt[stage], epsilon, B, context, stage, log_writer)
			if largeBlockID == 0 {
				cnn = bootstrapImaginaryPrint(cnn, context, 14, stage, log_writer)
			} else if largeBlockID == 1 {
				cnn = bootstrapImaginaryPrint(cnn, context, 13, stage, log_writer)
			} else if largeBlockID == 2 {
				cnn = bootstrapImaginaryPrint(cnn, context, 12, stage, log_writer)
			}
			cnn = approxReLUPrint(cnn, 13, log_writer, context, stage)

			stage = 2*((endNum+1)*largeBlockID+blockID) + 2
			fmt.Println("layer ", stage)
			st = 1

			cnn = compactGappedConvolutionPrint(cnn, co, st, fh, fw, convWgt[stage], bnVar[stage], bnWgt[stage], epsilon, context, stage, log_writer)
			cnn = compactGappedBatchNormPrint(cnn, bnBias[stage], bnMean[stage], bnVar[stage], bnWgt[stage], epsilon, B, context, stage, log_writer)
			if largeBlockID >= 1 && blockID == 0 {
				tempCnn = downsamplingPrint(tempCnn, context, stage, log_writer)
			}
			cnn = cipherAddPrint(cnn, tempCnn, context, stage, log_writer)
			if largeBlockID == 0 {
				cnn = bootstrapImaginaryPrint(cnn, context, 14, stage, log_writer)
			} else if largeBlockID == 1 {
				cnn = bootstrapImaginaryPrint(cnn, context, 13, stage, log_writer)
			} else if largeBlockID == 2 {
				cnn = bootstrapImaginaryPrint(cnn, context, 12, stage, log_writer)
			}
			cnn = approxReLUPrint(cnn, 13, log_writer, context, stage)
		}
	}
	cnn = averagepoolingPrint(cnn, B, context, log_writer)
	cnn = fullyConnectedPrint(cnn, linWgt, linBias, 10, 64, context, log_writer)

	elapse := time.Since(totalTimeStart)
	fmt.Printf("Done in %s \n", elapse)

	decryptPrintTxt(cnn.cipher_, log_writer, context, 10)

	rtnVec := make([]complex128, params.LogMaxSlots())
	encoder.Decode(decryptor.DecryptNew(cnn.cipher_), rtnVec)
	fmt.Printf("(")
	log_writer.WriteString("(")
	for i := 0; i < 9; i++ {
		fmt.Print(rtnVec[i], ", ")
		log_writer.WriteString(fmt.Sprintf("%6.10f", rtnVec[i]) + ", ")
	}
	fmt.Print(rtnVec[9], ")\n")
	log_writer.WriteString(fmt.Sprintf("%6.10f", rtnVec[9]) + ")\n")
	fmt.Println("total time: ", elapse)
	log_writer.WriteString("total time: " + elapse.String() + "\n")

	inferredLabel = 0
	maxScore = -100.0
	for i := 0; i < 10; i++ {
		if maxScore < real(rtnVec[i]) {
			inferredLabel = i
			maxScore = real(rtnVec[i])
		}
	}
	return inferredLabel, maxScore
}

func ImportParametersCifar10(layerNum int) (linwgt []float64, linbias []float64, convwgt [][]float64, bnbias [][]float64, bnmean [][]float64, bnvar [][]float64, bnwgt [][]float64, endNum int) {

	var dir string

	// directory name
	if layerNum != 20 && layerNum != 32 && layerNum != 44 && layerNum != 56 && layerNum != 110 {
		fmt.Println("layer number is not valid")
	}
	if layerNum == 20 {
		dir = "resnet20_new"
	} else if layerNum == 32 {
		dir = "resnet32_new"
	} else if layerNum == 44 {
		dir = "resnet44_new"
	} else if layerNum == 56 {
		dir = "resnet56_new"
	} else if layerNum == 110 {
		dir = "resnet110_new"
	} else {
		dir = ""
	}

	// endNum
	if layerNum == 20 {
		endNum = 2
	} else if layerNum == 32 {
		endNum = 4
	} else if layerNum == 44 {
		endNum = 6
	} else if layerNum == 56 {
		endNum = 8
	} else if layerNum == 110 {
		endNum = 17
	} else {
		fmt.Println("layer_num is not correct")
	}

	var num_c, num_b, num_m, num_v, num_w int
	// var num_c int

	convwgt = make([][]float64, layerNum-1)
	bnbias = make([][]float64, layerNum-1)
	bnmean = make([][]float64, layerNum-1)
	bnvar = make([][]float64, layerNum-1)
	bnwgt = make([][]float64, layerNum-1)

	fh, fw, ci, co := 3, 3, 0, 0

	// convolution parameters
	ci = 3
	co = 16
	ReadLinesIdx("parameters/resnet_pretrained/"+dir+"/conv1_weight.txt", convwgt, fh*fw*ci*co, num_c)
	num_c++

	// convolution parameters
	for j := 1; j <= 3; j++ {
		for k := 0; k <= endNum; k++ {
			// co setting
			if j == 1 {
				co = 16
			} else if j == 2 {
				co = 32
			} else if j == 3 {
				co = 64
			}

			// ci setting
			if j == 1 || (j == 2 && k == 0) {
				ci = 16
			} else if (j == 2 && k != 0) || (j == 3 && k == 0) {
				ci = 32
			} else {
				ci = 64
			}
			ReadLinesIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_conv1_weight.txt", convwgt, fh*fw*ci*co, num_c)
			num_c++

			// ci setting
			if j == 1 {
				ci = 16
			} else if j == 2 {
				ci = 32
			} else if j == 3 {
				ci = 64
			}
			ReadLinesIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_conv2_weight.txt", convwgt, fh*fw*ci*co, num_c)
			num_c++
		}
	}

	// batch_normalization parameters
	ci = 16
	ReadLinesIdx("parameters/resnet_pretrained/"+dir+"/bn1_bias.txt", bnbias, ci, num_b)
	num_b++
	ReadLinesIdx("parameters/resnet_pretrained/"+dir+"/bn1_running_mean.txt", bnmean, ci, num_m)
	num_m++
	ReadLinesIdx("parameters/resnet_pretrained/"+dir+"/bn1_running_var.txt", bnvar, ci, num_v)
	num_v++
	ReadLinesIdx("parameters/resnet_pretrained/"+dir+"/bn1_weight.txt", bnwgt, ci, num_w)
	num_w++

	// batch_normalization parameters
	for j := 1; j <= 3; j++ {
		if j == 1 {
			ci = 16
		} else if j == 2 {
			ci = 32
		} else if j == 3 {
			ci = 64
		}

		for k := 0; k <= endNum; k++ {
			ReadLinesIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn1_bias.txt", bnbias, ci, num_b)
			num_b++
			ReadLinesIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn1_running_mean.txt", bnmean, ci, num_m)
			num_m++
			ReadLinesIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn1_running_var.txt", bnvar, ci, num_v)
			num_v++
			ReadLinesIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn1_weight.txt", bnwgt, ci, num_w)
			num_w++
			ReadLinesIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn2_bias.txt", bnbias, ci, num_b)
			num_b++
			ReadLinesIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn2_running_mean.txt", bnmean, ci, num_m)
			num_m++
			ReadLinesIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn2_running_var.txt", bnvar, ci, num_v)
			num_v++
			ReadLinesIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn2_weight.txt", bnwgt, ci, num_w)
			num_w++
		}
	}
	// FC layer
	file, err := os.Open("parameters/resnet_pretrained/" + dir + "/linear_weight.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	in := bufio.NewScanner(file)
	linwgt = ReadLines(in, linwgt, 10*64)
	file, err = os.Open("parameters/resnet_pretrained/" + dir + "/linear_bias.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	in = bufio.NewScanner(file)
	linbias = ReadLines(in, linbias, 10)

	return linwgt, linbias, convwgt, bnbias, bnmean, bnvar, bnwgt, endNum
}

func DecryptPrint(params hefloat.Parameters, ciphertext *rlwe.Ciphertext, decryptor rlwe.Decryptor, encoder hefloat.Encoder) {

	n := 1 << ciphertext.LogSlots()
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

func ReadLinesIdx(fileName string, storeVal [][]float64, lineNum int, idx int) {
	in, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer in.Close()

	scanner := bufio.NewScanner(in)
	for i := 0; i < lineNum; i++ {
		scanner.Scan()
		s := scanner.Text()
		p, _ := strconv.ParseFloat(s, 64)
		// fmt.Println(p)
		storeVal[idx] = append(storeVal[idx], p)
	}
}
func SkipLines(scanner *bufio.Scanner, lineNum int) {
	for i := 0; i < lineNum; i++ {
		scanner.Scan()
		scanner.Text()
	}
}

func ReadLines(scanner *bufio.Scanner, storeVal []float64, lineNum int) []float64 {
	for i := 0; i < lineNum; i++ {
		scanner.Scan()
		s := scanner.Text()
		p, _ := strconv.ParseFloat(s, 64)
		storeVal = append(storeVal, p)
	}

	return storeVal
}

func ReadLinesInt(scanner *bufio.Scanner, storeVal []int, lineNum int) []int {
	for i := 0; i < lineNum; i++ {
		scanner.Scan()
		s := scanner.Text()
		p, _ := strconv.Atoi(s)
		storeVal = append(storeVal, p)
	}

	return storeVal
}
