package comp

import (
	"bufio"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/lifejade/cnn_ckksv5/cnn/scaleinv"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
)

func Pow2(n int) (prod int) {

	prod = 1
	for i := 0; i < n; i++ {
		prod *= 2
	}

	return prod
}
func ReLU(x float64) (y float64) {
	if x > 0 {
		y = x
	} else {
		y = 0.0
	}

	return y
}
func NumOne(n int) (num int) {

	i, num := 0, 0
	for {
		if Pow2(i) > n {
			break
		}
		i++
	}
	for j := 0; j < i; j++ {
		if n%2 == 1 {
			num++
		}
		n /= 2
	}
	return num
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
func ReadLinesComplex(scanner *bufio.Scanner, storeVal []complex128, lineNum int) []complex128 {
	for i := 0; i < lineNum; i++ {
		scanner.Scan()
		s := scanner.Text()
		p, _ := parseComplex(s)
		storeVal = append(storeVal, p)
	}

	return storeVal
}
func parseComplex(s string) (complex128, error) {
	s = strings.Trim(s, "()")      // remove parentheses
	parts := strings.Split(s, "-") // split real and imaginary parts
	if len(parts) != 2 {
		return 0, fmt.Errorf("invalid format")
	}

	realStr := parts[0]                          // real part
	imagStr := strings.TrimSuffix(parts[1], "i") // imaginary part

	real, err := strconv.ParseFloat(realStr, 64) // parse real part
	if err != nil {
		return 0, err
	}

	imag, err := strconv.ParseFloat(imagStr, 64) // parse imaginary part
	if err != nil {
		return 0, err
	}

	return complex(real, imag), nil
}
func ShowFailureReLU(cipher *rlwe.Ciphertext, x []complex128, precision int, context scaleinv.ScaleContext) (failure int) {

	logn := context.Params_.LogN() - 1
	n := 1 << logn

	encoder := *context.Encoder_
	decryptor := *context.Decryptor_

	failure = 0
	bound := math.Pow(2.0, float64(-precision))

	output := make([]complex128, n)
	err := encoder.Decode(decryptor.DecryptNew(cipher), output)
	scaleinv.PrintErr(err)
	for i := 0; i < 1<<logn; i++ {
		if math.Abs(ReLU(real(x[i]))-real(output[i])) > bound {
			failure++
		}
	}
	fmt.Println("-------------------------------------------------")
	fmt.Println("failure : ", failure)
	fmt.Println("-------------------------------------------------")

	return failure

}
