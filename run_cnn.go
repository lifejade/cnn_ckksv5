package main

import (
	"fmt"

	"github.com/lifejade/cnn_ckksv5/cnn"
)

func main() {
	var startImageID int

	fmt.Print("Start image ID: ")
	fmt.Scanln(&startImageID)
	cnn.ResNetCifar10(50, startImageID)
}
