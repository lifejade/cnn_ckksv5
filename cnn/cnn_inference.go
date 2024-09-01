package cnn

import (
	"fmt"
	"runtime"
	"sync"

	"bufio"
	"log"
	"os"
	"strconv"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"github.com/tuneinsight/lattigo/v5/utils"
)

func ResNetCifar10(layerNum int, imageID int) {
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
	SkipLines(in, 32*32*3*imageID)
	var image []float64
	image = ReadLines(in, image, 32*32*3)

	//result file init
	var result_file *os.File
	result_file, _ = os.Create("result/resnet" + strconv.Itoa(layerNum) + "_cifar10_label" + strconv.Itoa(imageID) + ".txt")
	outShare := bufio.NewWriter(result_file)
	_ = outShare
	fmt.Println("Resnet-CKKS result name: ", result_file.Name())

	//ckks parameter init
	ckksParams := CNN_Cifar10_Parameters
	params, err := hefloat.NewParametersFromLiteral(ckksParams.SchemeParams)
	if err != nil {
		panic(err)
	}
	fmt.Println("ckks parameter init end")

	//boot parameter init
	bootParams14, bootParams13, bootParams12 := ckksParams.BootstrappingParams, ckksParams.BootstrappingParams, ckksParams.BootstrappingParams
	//Todo : not RS-Bootstrapping yet
	bootParams14.LogSlots, bootParams13.LogSlots, bootParams12.LogSlots = utils.Pointy(15), utils.Pointy(15), utils.Pointy(15)
	btpParams14, err := bootstrapping.NewParametersFromLiteral(params, bootParams14)
	if err != nil {
		panic(err)
	}
	btpParams13, err := bootstrapping.NewParametersFromLiteral(params, bootParams13)
	if err != nil {
		panic(err)
	}
	btpParams12, err := bootstrapping.NewParametersFromLiteral(params, bootParams12)
	if err != nil {
		panic(err)
	}
	fmt.Println("bootstrapping parameter init end")
	// generate keys
	//fmt.Println("generate keys")
	//keytime := time.Now()
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
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
	//ellapse := time.Since(keytime)
	//fmt.Printf("GaloisElement time end : %s\n", ellapse)

	rlk := kgen.GenRelinearizationKeyNew(sk)
	var rtk []*rlwe.GaloisKey = make([]*rlwe.GaloisKey, len(galEls))
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
	//rtk = kgen.GenGaloisKeysNew(galEls, sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	fmt.Println(evk.GaloisKeys)
	//ellapse = time.Since(keytime)
	//fmt.Println("generate key end")
	//fmt.Printf("time end : %s\n", ellapse)

	//generate -or/er
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)
	fmt.Println("generate Evaluator end")

	_, _, _, _ = encryptor, decryptor, encoder, evaluator

	//generate bootstrapper
	btpevk14, _, _ := btpParams14.GenEvaluationKeys(sk)
	btp14, _ := bootstrapping.NewEvaluator(btpParams14, btpevk14)
	btpevk13, _, _ := btpParams13.GenEvaluationKeys(sk)
	btp13, _ := bootstrapping.NewEvaluator(btpParams13, btpevk13)
	btpevk12, _, _ := btpParams12.GenEvaluationKeys(sk)
	btp12, _ := bootstrapping.NewEvaluator(btpParams12, btpevk12)
	_, _, _ = btp14, btp13, btp12

	fmt.Println("generated bootstrapper end")

	//Resnet parameter init
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
	_, _, _, _ = co, st, fh, fw
	_, _ = logp, logq
	_, _ = stage, epsilon
	_, _, _, _, _, _, _, _ = linWgt, linBias, convWgt, bnBias, bnMean, bnVar, bnWgt, endNum

	//preprocess inference image
	for len(image) < n {
		image = append(image, 0.0)
	}
	for i := n / init_p; i < n; i++ { // copy and paste
		image[i] = image[i%(n/init_p)]
	}
	for i := 0; i < n; i++ {
		image[i] /= B
	}
	context := NewContext(encoder, encryptor, decryptor, sk, pk, btp14, btp13, btp12, rtk, rlk, evaluator, &params)

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
	ct := sumSlot(cipher, 6, 1, context)
	DecryptPrint(params, ct, *decryptor, *encoder)

	cnn := NewTensorCipherFormData(1, 32, 32, 3, 3, init_p, logn, image, context)
	fmt.Println("preprocess & encrypt inference image end")

	cnn = compactGappedConvolution(cnn, co, st, fh, fw, convWgt[0], bnVar[0], bnWgt[0], epsilon, context)
	_ = cnn
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
	ReadLinesIdx("pretrained_parameters/"+dir+"/conv1_weight.txt", convwgt, fh*fw*ci*co, num_c)
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
			ReadLinesIdx("pretrained_parameters/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_conv1_weight.txt", convwgt, fh*fw*ci*co, num_c)
			num_c++

			// ci setting
			if j == 1 {
				ci = 16
			} else if j == 2 {
				ci = 32
			} else if j == 3 {
				ci = 64
			}
			ReadLinesIdx("pretrained_parameters/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_conv2_weight.txt", convwgt, fh*fw*ci*co, num_c)
			num_c++
		}
	}

	// batch_normalization parameters
	ci = 16
	ReadLinesIdx("pretrained_parameters/"+dir+"/bn1_bias.txt", bnbias, ci, num_b)
	num_b++
	ReadLinesIdx("pretrained_parameters/"+dir+"/bn1_running_mean.txt", bnmean, ci, num_m)
	num_m++
	ReadLinesIdx("pretrained_parameters/"+dir+"/bn1_running_var.txt", bnvar, ci, num_v)
	num_v++
	ReadLinesIdx("pretrained_parameters/"+dir+"/bn1_weight.txt", bnwgt, ci, num_w)
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
			ReadLinesIdx("pretrained_parameters/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn1_bias.txt", bnbias, ci, num_b)
			num_b++
			ReadLinesIdx("pretrained_parameters/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn1_running_mean.txt", bnmean, ci, num_m)
			num_m++
			ReadLinesIdx("pretrained_parameters/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn1_running_var.txt", bnvar, ci, num_v)
			num_v++
			ReadLinesIdx("pretrained_parameters/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn1_weight.txt", bnwgt, ci, num_w)
			num_w++
			ReadLinesIdx("pretrained_parameters/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn2_bias.txt", bnbias, ci, num_b)
			num_b++
			ReadLinesIdx("pretrained_parameters/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn2_running_mean.txt", bnmean, ci, num_m)
			num_m++
			ReadLinesIdx("pretrained_parameters/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn2_running_var.txt", bnvar, ci, num_v)
			num_v++
			ReadLinesIdx("pretrained_parameters/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn2_weight.txt", bnwgt, ci, num_w)
			num_w++
		}
	}
	// FC layer
	file, err := os.Open("pretrained_parameters/" + dir + "/linear_weight.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	in := bufio.NewScanner(file)
	linwgt = ReadLines(in, linwgt, 10*64)
	file, err = os.Open("pretrained_parameters/" + dir + "/linear_bias.txt")
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

	return

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
