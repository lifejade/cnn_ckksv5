package main

import (
	"fmt"

	"github.com/lifejade/cnn_ckksv5/cnn"
)

func main() {
	//var layerNum, startImageID int

	fmt.Print("Layer Num(20) : ")
	//fmt.Scan(&layerNum)
	fmt.Print("Start image ID: ")
	//fmt.Scanln(&startImageID)
	cnn.ResNetCifar10(20, 0)
}
