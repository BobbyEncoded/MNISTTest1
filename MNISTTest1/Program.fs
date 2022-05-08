open System.IO

module ArrayOperations =
    let multiply2DArrayByScalar (inputArray : float32[,]) (scale : float32) = Array2D.map (fun e -> e * scale) inputArray
    let Transpose (inputArray : 'T[,]) = Array2D.init (Array2D.length2 inputArray) (Array2D.length1 inputArray) (fun row col -> (col, row) ||> Array2D.get inputArray)
    let DotProduct2D (array1 : float32[,]) (array2 : float32[,]) =
        match (Array2D.length2(array1) = Array2D.length1(array2)) with
        | false -> raise (System.ArgumentException("Array lengths did not match"))
        | true ->
            let finalHeight = Array2D.length1(array1)
            let finalWidth = Array2D.length2(array2)
            Array2D.init finalHeight finalWidth (fun rowCoord colCoord ->
                let cols = array1[rowCoord, *]
                let rows = array2[*, colCoord]
                (rows, cols)
                ||> Array.map2 (fun a b -> a * b)
                |> Array.sum)
    let ArrayAdd2D (array1 : float32[,]) (array2 : float32[,]) =
        match ((Array2D.length1 array1 = Array2D.length1 array2) && (Array2D.length2 array1 = Array2D.length2 array2)) with
        | false -> raise (System.ArgumentException("Array addition performed when arrays were not the same size"))
        | true -> Array2D.init (Array2D.length1 array1) (Array2D.length2 array2) (fun row col -> (Array2D.get array1 row col) + (Array2D.get array2 row col) )
    let Turn1DArrayInto2DVertical (array : 'T[]) = Array2D.init (array |> Array.length) (1) (fun row _ -> Array.get array row)
    let Turn1DArrayInto2DHorizontal (array : 'T[]) = Array2D.init (1) (array |> Array.length) (fun _ col -> Array.get array col)

module MachineLearningHelpers =
    let Sigmoid (x : float32) = 1.0f / (1.0f + (float32 (System.Math.Exp(float -x))))
    let SigmoidArray (x : float32[]) = Array.map Sigmoid x
    let Sigmoid2DArray (x : float32[,]) = Array2D.map Sigmoid x
    let Sigmoid' (x : float32) = (Sigmoid x) * (1.0f - (Sigmoid x))
    let Sigmoid'Array (x : float32[]) = Array.map Sigmoid' x
    let Sigmoid'2DArray (x : float32[,]) = Array2D.map Sigmoid' x
    let Sigmoid'FromSigmoid (x : float32) = x * (1.0f - x)
    let Sigmoid'FromSigmoidArray (x : float32[]) = Array.map Sigmoid'FromSigmoid x
    let Sigmoid'FromSigmoid2DArray (x : float32[,]) = Array2D.map Sigmoid'FromSigmoid x
    let ReLu (x : float32) = if x <= 0.0f then 0.0f else x
    let ReLuArray (x : float32[]) = Array.map ReLu x
    let ReLu2DArray (x : float32[,]) = Array2D.map ReLu x
    let ReLu' (x : float32) = if x <= 0.0f then 0.0f else 1.0f
    let ReLu'Array (x : float32[]) = Array.map ReLu' x
    let ReLu'2DArray (x : float32[,]) = Array2D.map ReLu' x

type DigitImage =
    {
        Label: byte;
        Data: byte[,]
    }

type BoundingFunction =
    | ReLu //Rectified Linear Unit
    | Sigmoid

type TestMode =
    | Test
    | Production

module MNIST =
    let CollectImages (imageFileAddress : string) (labelFileAddress : string) : Result<DigitImage[], string> =
            //Create immutable file stream options
            let FileStreamOptions =
                let mutable testImageStreamOptions : FileStreamOptions = new FileStreamOptions()
                testImageStreamOptions.Access <- FileAccess.Read //We only want to read the file, not write to it
                testImageStreamOptions.Mode <- FileMode.Open //The file already exists and we want to open it
                testImageStreamOptions.Options <- FileOptions.SequentialScan //We are going to go through the file from beginning to end.
                testImageStreamOptions

            //Establish how to get a binary reader for a file
            let getBinaryReader address =
                new BinaryReader(new FileStream(address, FileStreamOptions))

            //Create the binary readers to read the data
            use ImageStream = getBinaryReader imageFileAddress
            use LabelStream = getBinaryReader labelFileAddress

            //Converts the MNIST integers into .NET Integers.  MNIST data is Big endian, so we must perform a bit shift of 8 bits (1 byte) to organize the data properly
            let readInt (b: BinaryReader) = 
                (0, [1..4]) ||> List.fold (fun res _ -> (res <<< 8) ||| (int)(b.ReadByte()))

            let readByte (b: BinaryReader) =
                b.ReadByte()

            let isMagicNumber = readInt(ImageStream)
            let isNImages = readInt(ImageStream)
            let isNRows = readInt(ImageStream)
            let isNColumns = readInt(ImageStream)

            printfn "Image Stream: Magic Number: %i, Num Images: %i, Num Rows: %i, Num Cols: %i" isMagicNumber isNImages isNRows isNColumns

            let lsMagicNumber = readInt(LabelStream)
            let lsNItems = readInt(LabelStream)

            printfn "Label Stream: Magic Number: %i, Num Items: %i" lsMagicNumber lsNItems

            match (isNImages = lsNItems) with
            | false -> Result.Error "Number of images did not match number of labels"
            | true -> 
                let nItems = isNImages
                //Now we need to align the labels and the images together.  We need to grab the label by reading the label byte, and read the data by reading the data bytes.
                let createDigitImage numRows numCols : DigitImage =
                    let mutable array = Array2D.create numCols numRows 0uy
                    for y in 0 .. (isNRows-1) do
                        for x in 0 .. (isNColumns-1) do
                            ImageStream
                            |> readByte
                            |> Array2D.set array x y
                    let label () = LabelStream |> readByte
                    {Label = label (); Data = array}
                let digitImages numImages =
                    seq {0 .. (numImages-1)}
                    |> Seq.map (fun _ -> createDigitImage isNRows isNColumns)
                    |> Array.ofSeq
                let images = digitImages nItems
                Result.Ok images
    let createImageFromData (fileName : string) (inputImageData : byte[,]) : unit =
            let numCols = inputImageData |> Array2D.length2
            let numRows = inputImageData |> Array2D.length1
            //Get the address of the pinned array element
            let nativeint = System.Runtime.InteropServices.Marshal.UnsafeAddrOfPinnedArrayElement(inputImageData,0)
            let mutable bitmap : System.Drawing.Bitmap = new System.Drawing.Bitmap(numCols, numRows, numCols, System.Drawing.Imaging.PixelFormat.Format8bppIndexed, nativeint)
            let mutable palette = bitmap.Palette
            let mutable entries = palette.Entries
            for i in 0 .. 255 do
                let color = System.Drawing.Color.FromArgb(i, i, i)
                entries[i] <- color
            bitmap.Palette <- palette
            bitmap.Save(fileName)
    //The data comes in as bytes, but all the neural network math will be done on float32s.  Furthermore, we need a vector rather than a 2d array to initialize the neural network, as the layers are represented as vectors, and this will become the initial layer.
    let convertDataToFloatVector (inputData : byte[,]) : float32[] =
        inputData
        |> Array2D.map float32
        |> System.Linq.Enumerable.Cast<float32>
        |> Array.ofSeq

module DigitImage =
    let ConvertDigitImageToFloats (inputImage : DigitImage) =
        let inputData = inputImage.Data |> MNIST.convertDataToFloatVector
        let expectedOutputs = Array.init 10 (fun i -> if i = (int inputImage.Label) then 1.0f else 0.0f)
        (inputData, expectedOutputs)

type Layer(weights : float32[,], biases : float32[], boundingFunction : BoundingFunction) =
    //Class used to verify that weights and biases are the same and in a valid arrangement on initialization
    do
        let wRows = Array2D.length1(weights)
        let bRows = Array.length(biases)
        match (bRows = wRows) with
        | false -> raise (System.ArgumentException("Number of rows in weights and biases did not match"))
        | true -> ()
    member this.Weights = weights
    member this.Biases = biases
    member this.BoundingFunction = boundingFunction
    static member createRandomLayer (height : int) (width : int) =
        let random = System.Random ()
        let randomizedWeightsArray : float32[,] = Array2D.zeroCreate height width |> Array2D.map(fun _ -> random.NextDouble() |> float32 |> (+) -0.5f |> (*) 2.0f) //Want weights between -1 and +1
        let biasArray = Array.zeroCreate(height) |> Array.map (fun _ -> random.NextDouble() |> float32 |> (+) -0.5f |> (*) 1.0f) //Want biases between -1 and +1
        (randomizedWeightsArray, biasArray)

type NeuralNetwork(hiddenLayers : Layer[]) =
    //Class used to verify the neural network is a structure composed of valid layers on initialization
    do
        let rec checkLayers (prevLayerHeight : int option) (remainingLayers : Layer[]) =
            let currentLayer = remainingLayers |> Array.tryHead
            match currentLayer with
            | None -> ()
            | Some layer ->
                let nextLayers = remainingLayers |> Array.tail
                match prevLayerHeight with
                | None -> checkLayers (Some(Array2D.length1(layer.Weights))) (nextLayers)
                | Some prevLayerHeight ->
                    let currentLayerWidth = Array2D.length2(layer.Weights)
                    match (currentLayerWidth = prevLayerHeight) with
                    | false -> raise (System.ArgumentException("Neural Network layer map was invalid. One of the layers widths did not match the previous layer's height."))
                    | true -> checkLayers (Some(Array2D.length1(layer.Weights))) (nextLayers)
        checkLayers None hiddenLayers
    member this.Network = hiddenLayers
    //This can be used to verify the first network layer
    member this.inputLayerLength () = this.Network |> Array.head |> (fun l -> l.Weights |> Array2D.length2) //This tell us the layer allowed as the input to this neural network by checking what it is expecting as an input
    member this.FeedForward (inputVector : float32[]) =
        let feedOneLayer (inputVector : float32[]) (currentLayer : Layer) = //This returns each current state at the time, for purposes of backpropagation
            let (%*) = ArrayOperations.DotProduct2D
            let (%+) = ArrayOperations.ArrayAdd2D
            let weights = currentLayer.Weights
            let biases = currentLayer.Biases
            let algorithm = currentLayer.BoundingFunction
            let turnVectorVertical2D inputVector = Array2D.init (inputVector |> Array.length) 1 (fun row _ -> row |> Array.get inputVector)
            let weightedLayer = weights %* (turnVectorVertical2D inputVector)
            let biasedLayer = weightedLayer %+ (turnVectorVertical2D biases)
            let resultVector = biasedLayer |> System.Linq.Enumerable.Cast<float32> |> Array.ofSeq
            match algorithm with
            | ReLu ->
                let finalResult = resultVector |> MachineLearningHelpers.ReLuArray
                (finalResult, finalResult)
            | Sigmoid ->
                let finalResult = resultVector |> MachineLearningHelpers.SigmoidArray
                (finalResult, finalResult)
        //The Result in this case is the mapping of all layers to their activated layer form (That is, it transforms each element in the array into what that layer's final activations looked like.)   The final state is the end result of the network propogation.
        let activations, finalResult =
            (inputVector, hiddenLayers)
            ||> Array.mapFold feedOneLayer
        (Array.append (Array.singleton inputVector) activations), finalResult //The input vector is the activation of the 0th layer, in essence.  We need to append it back on.
    //Creates a randomized neural network when it knows how many nodes are in layer 0 and then a count of the nodes in the following layers. Layer 0 is updated only at runtime with new input information.
    static member createRandomizedNeuralNetwork (initialLayerHeight : int) (sizeOfEachLayer : int[]) : NeuralNetwork =
        let firstLayerSize = sizeOfEachLayer |> Array.head
        let remainingLayerSizes = sizeOfEachLayer |> Array.tail
        let layer1 =
            let (weights, biases) = Layer.createRandomLayer firstLayerSize initialLayerHeight
            Layer(weights, biases, BoundingFunction.ReLu)
        let layerCreationFoldingFunction (previousLayers : Layer[]) (currentNodeSizes : int) = 
            let lastLayer = previousLayers |> Array.last
            let width = lastLayer.Weights |> Array2D.length1
            let (weights, biases) = Layer.createRandomLayer currentNodeSizes width
            let nextLayer = Layer(weights, biases, BoundingFunction.ReLu) |> Array.singleton
            Array.append previousLayers nextLayer
        (layer1 |> Array.singleton, remainingLayerSizes)
        ||> Array.fold layerCreationFoldingFunction
        |> (fun l ->
            let lastLayer = l |> Array.last
            let newFinalLayer = Layer(lastLayer.Weights, lastLayer.Biases, BoundingFunction.Sigmoid)
            l |> Array.updateAt ((l |> Array.length) - 1) newFinalLayer
            )
        |> NeuralNetwork
    static member feedForward (inputNetwork : NeuralNetwork) (inputVector : float32[]) = inputNetwork.FeedForward(inputVector)
    static member backprop (inputNetwork : NeuralNetwork) (inputData : float32[]) (expectedResult : float32[]) = //Returns set of new weight derivatives for the data run
        let (activations, finalState) = inputData |> NeuralNetwork.feedForward inputNetwork
        let finalArrayLength = Array.length finalState //Precalculated for performance reasons.
        if (finalArrayLength <> (Array.length expectedResult)) then raise (System.ArgumentException("Output and Expected Result lengths did not match during backpropagation."))
        let errorFunction (inputResult : float32) (expectedResult : float32) = -2.0f * (expectedResult - inputResult)// ** 2.0f //Exponential is not necessary since this is the derivative.
        let finalLayerError = (finalState, expectedResult) ||> Array.map2 errorFunction
        //Note that all activation functions we are using have derivatives that can be defined in terms of the original activation function, not just in terms of the variable which went into the activation function.
        //We will be using that so that we do not have to store the data on the inputs of the activation function.
        let activationFunctionDerivativeToUseBasedOnActivatedNeuron (boundingFunction : BoundingFunction) =
            match boundingFunction with
            | ReLu -> MachineLearningHelpers.ReLu'
            | Sigmoid -> MachineLearningHelpers.Sigmoid'FromSigmoid
        let finalLayerActivationFunction = (inputNetwork.Network |> Array.last).BoundingFunction |> activationFunctionDerivativeToUseBasedOnActivatedNeuron
        let finalLayerPreDeltas = finalLayerError |> Array.map (fun e -> e * (1.0f / (float32 finalArrayLength))) //Perform scaling of the errors on all array elements
        let finalLayerActivationDerivatives = finalState |> Array.map finalLayerActivationFunction //Get all the derivatives of the activation functions
        let finalLayerDeltas = (finalLayerActivationDerivatives, finalLayerPreDeltas) ||> Array.map2 (*) //Multiply the two together.  This is the delta of the Final Layer, due to chain rule.  Delta is not a derivative, it's just a precalculated constant.
        let derivativeWeightMatrix = ArrayOperations.DotProduct2D (finalLayerDeltas |> ArrayOperations.Turn1DArrayInto2DVertical) ((activations[activations.Length-2]) |> ArrayOperations.Turn1DArrayInto2DHorizontal) //When a vertical delta matrix is multiplied by the inputs from the previous layer, we get the matrix which tells us how to adjust weights.
        let derivativeBiasMatrix = finalLayerDeltas //The bias matrix is the weight matrix without dot multiplying the activations.  This is just the deltas.
        //Now we have all the information for the final layer of derivatives and deltas.
        
        let rec backpropOfOtherLayers (currentLayer : int) (currentWeightAndBiasDerivatives : (float32[,] * float32[])[]) (previousLayerDelta : float32[]) (neuralNetworkStructure : NeuralNetwork) (activationsOfAllLayers : float32[][]) =
            match currentLayer with
            | currentLayer when currentLayer <= 0 -> currentWeightAndBiasDerivatives |> Array.rev
            | _ ->
                let activationFunctionOfLayer = activationFunctionDerivativeToUseBasedOnActivatedNeuron neuralNetworkStructure.Network[currentLayer-1].BoundingFunction
                let lastLayersActivations = (currentLayer) |> Array.get activationsOfAllLayers
                let thisLayersActivations = (currentLayer - 1) |> Array.get activationsOfAllLayers
                let preDeltaComputation2D = ArrayOperations.DotProduct2D  (previousLayerDelta |> ArrayOperations.Turn1DArrayInto2DHorizontal) (neuralNetworkStructure.Network[currentLayer].Weights)  //Dot multiply horizontal deltas from last layer by weight matrix (also from last last layer, because the weights and deltas mixed to become our new delta terms) to get horizontal preDeltas
                let preDeltaComputation = preDeltaComputation2D[0,*]
                let deltaThisLayer = (preDeltaComputation, (lastLayersActivations |> Array.map activationFunctionOfLayer)) ||> Array.map2 (*) //Multiply that horizontal array piecewise with the derivatives of the activation function to get the new deltas.
                let dCostWithRespectTodWeights = ArrayOperations.DotProduct2D (deltaThisLayer |> ArrayOperations.Turn1DArrayInto2DVertical) (thisLayersActivations |> ArrayOperations.Turn1DArrayInto2DHorizontal)
                let dCostWithRespectTodBiases = deltaThisLayer //Once again biases are equivalent to deltas
                let thisLayerWeightAndBiasDerivatives =  ((dCostWithRespectTodWeights, dCostWithRespectTodBiases) |> Array.singleton)
                let newWeightAndBiasDerivatives = Array.append currentWeightAndBiasDerivatives thisLayerWeightAndBiasDerivatives
                backpropOfOtherLayers (currentLayer-1) (newWeightAndBiasDerivatives) (deltaThisLayer) (neuralNetworkStructure) (activationsOfAllLayers)
        backpropOfOtherLayers (inputNetwork.Network |> Array.length |> (+) -1) (Array.zip (derivativeWeightMatrix |> Array.singleton) (derivativeBiasMatrix |> Array.singleton)) (finalLayerDeltas) (inputNetwork) activations
        //Note that the final layer deltas are turned horizontal.  The previous layer's deltas are horizontal and multiplied by the weights of the current layer and also multiplied by the derivatives of the activation functions piecewise to get this layer's deltas, horizontal.

    //SGD stands for Stochastic Gradient Descent.  Training data is all the data required for the network, where the first element is the input data in and the second element is the expected output arrangement (probably an array of 0s with 1 1).  That itself is an array.  The learning rate is represented by eta, and is a multiplier to speed up learning.
    static member SGD (trainingData : (float32[] * float32[])[]) (epochs : int) (miniBatchSize : int) (learningRate : float32) (testMode : TestMode) (initialNetwork : NeuralNetwork) =
        let random = System.Random ()
        let testNeuralNet (neuralNetToTest : NeuralNetwork) =
                let numCorrectResults = 
                    trainingData
                    |> Array.Parallel.map (fun (input, expectedOutput) ->
                        let currentResults = snd(neuralNetToTest.FeedForward(input))
                        let findGreatestIteration (inputArray : float32[]) =
                            let maxValue = inputArray |> Array.max
                            inputArray |> Array.findIndex(fun v -> v = maxValue)
                        let mostLikelyResult = findGreatestIteration currentResults
                        let expectedResult = findGreatestIteration expectedOutput
                        match (mostLikelyResult = expectedResult) with
                        | false -> 0
                        | true -> 1
                        )
                    |> Array.sum
                let numResultsTotal = trainingData |> Array.length
                let percentCorrect = (float32 numCorrectResults) / (float32 numResultsTotal)
                printfn "Percent Correct: %f, Num Correct Found: %i, Total Number of Images: %i" percentCorrect numCorrectResults numResultsTotal
        if (testMode = TestMode.Test) then testNeuralNet initialNetwork
        let rec completeAllEpochs (testMode : TestMode) (currentCount : int) (currentNetwork : NeuralNetwork) =
            match (currentCount < epochs) with
            | false -> currentNetwork
            | true ->
                let randomizedTrainingData = trainingData |> Array.sortBy (fun f -> random.Next())
                let miniBatches = randomizedTrainingData |> Array.chunkBySize miniBatchSize // The mini batches is just the training data broken into size chunks which are of size miniBatchSize.  At the end of each mini batch, we'll calculate a gradient and apply it to new weights and biases, and run the next batch.
                let updateOnBatch (learningRate : float32) (currentNeuralNetwork : NeuralNetwork) (trainingData : (float32[] * float32[])[]) =
                    let runBackpropOnBatch = trainingData |> Array.Parallel.map (fun (inputData, expectedResult) -> NeuralNetwork.backprop currentNeuralNetwork inputData expectedResult)
                    let averageResultsForUpdate =
                        let inverseNumResults = (1.0f / float32(runBackpropOnBatch |> Array.length))
                        let summedResults =
                            runBackpropOnBatch
                            |> Array.reduce (fun prevDerivArray nextDerivArray ->
                                (prevDerivArray, nextDerivArray)
                                ||> Array.map2 (fun (prevWeights, prevBiases) (nextWeights, nextBiases) -> ((ArrayOperations.ArrayAdd2D prevWeights nextWeights), ((prevBiases, nextBiases) ||> Array.map2 (fun prev next -> prev + next)))))
                        summedResults |> Array.map (fun (weights, biases) -> ((weights |> Array2D.map (fun e -> e * inverseNumResults * learningRate)), biases |> Array.map (fun e -> e * inverseNumResults * learningRate)))
                    //Remember to SUBTRACT the derivatives away from the weights to reduce the cost function
                    let updatedHiddenLayers = 
                        let (oldWeights, oldBiases) = currentNeuralNetwork.Network |> Array.map (fun e -> (e.Weights, e.Biases)) |> Array.unzip
                        let (negDerivWeights, negDerivBiases) = averageResultsForUpdate |> Array.map (fun (weights, biases) -> ((Array2D.map (fun i -> -i) weights), (Array.map (fun i -> -i) biases))) |> Array.unzip //Here we perform the switch to negative derivatives, to descend the gradient
                        let newWeights = (oldWeights, negDerivWeights) ||> Array.map2 (fun oldWeightLayer negDerivWeightLayer -> ArrayOperations.ArrayAdd2D oldWeightLayer negDerivWeightLayer)
                        let newBiases = (oldBiases, negDerivBiases) ||> Array.map2 (fun oldBiasLayer negDerivBiasLayer -> (oldBiasLayer, negDerivBiasLayer) ||> Array.map2 (fun o n -> o + n))
                        let newLayers = currentNeuralNetwork.Network |> Array.mapi (fun iter layer -> Layer(newWeights[iter], newBiases[iter], layer.BoundingFunction))
                        newLayers
                    NeuralNetwork(updatedHiddenLayers)
                let endOfEpochNeuralNet = (currentNetwork, miniBatches) ||> Array.fold (updateOnBatch learningRate)
                match testMode with
                | Production -> completeAllEpochs testMode (currentCount+1) endOfEpochNeuralNet
                | Test ->
                    testNeuralNet endOfEpochNeuralNet
                    completeAllEpochs testMode (currentCount+1) endOfEpochNeuralNet
        //In each epoch
        //Shuffle the training data
        //Split it into a bunch of mini batches randomly
        //In each mini batch, update the mini batch by performing backprop with learning factor eta
        //If we have testMode true, then print the results of each epoch
        //Else just print when the last epoch is complete.
        completeAllEpochs testMode 0 initialNetwork
    
    
// TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
// [offset] [type]          [value]          [description] 
// 0000     32 bit integer  0x00000803(2051) magic number 
// 0004     32 bit integer  60000            number of images 
// 0008     32 bit integer  28               number of rows 
// 0012     32 bit integer  28               number of columns 
// 0016     unsigned byte   ??               pixel 
// 0017     unsigned byte   ??               pixel 
// ........ 
// xxxx     unsigned byte   ??               pixel

// TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
// [offset] [type]          [value]          [description] 
// 0000     32 bit integer  0x00000803(2051) magic number 
// 0004     32 bit integer  10000            number of images 
// 0008     32 bit integer  28               number of rows 
// 0012     32 bit integer  28               number of columns 
// 0016     unsigned byte   ??               pixel 
// 0017     unsigned byte   ??               pixel 
// ........ 
// xxxx     unsigned byte   ??               pixel

module Main =
    [<EntryPointAttribute>]
    let main argv =
        let (&*) (array : float32[,]) (scalar : float32) = array |> Array2D.map(fun e -> e + scalar)
        let (&+) (offset : float32) (inputMatrix : float32[,]) = inputMatrix |> Array2D.map (fun v -> v + offset)
        let (%*) = ArrayOperations.DotProduct2D
        let (@+) (array1 : float[]) (array2 : float[]) = (array1, array2) ||> Array.map2 (fun a b -> a + b)

        let testImages = MNIST.CollectImages @"C:\Users\bobma\Downloads\MNIST\gzip\emnist-mnist-test-images-idx3-ubyte.bin" @"C:\Users\bobma\Downloads\MNIST\gzip\emnist-mnist-test-labels-idx1-ubyte.bin"
        let trainImages = MNIST.CollectImages @"C:\Users\bobma\Downloads\MNIST\gzip\emnist-mnist-train-images-idx3-ubyte.bin" @"C:\Users\bobma\Downloads\MNIST\gzip\emnist-mnist-train-labels-idx1-ubyte.bin"

        let image =
            let firstImage : DigitImage =
                match testImages with
                | Result.Error errorString -> raise (new InvalidDataException(errorString))
                | Result.Ok dataArray -> Array.get dataArray 3
            printfn "Label: %u" firstImage.Label
            firstImage.Data
        
        image
        |> MNIST.createImageFromData @"C:\Users\bobma\Pictures\MNIST Bitmaps\image.bmp"
        |> ignore

        let goodImages =
            match trainImages with
            | Result.Error _ -> raise (InvalidDataException("Image creation failed"))
            | Result.Ok images -> images

        //This gets the value for how many nodes will be in Layer 0, which we use to find the width of the weights of Layer 1
        let initialLayerHeight =
            let firstImage = goodImages |> Array.head
            let width = firstImage.Data |> Array2D.length2
            let height = firstImage.Data |> Array2D.length1
            width * height

        let testPropNeuralNet =
            let weights01vector = [|[|0.3f; 0.1f|];[|-0.4f;0.7f|]|]
            let weights01 = Array2D.init 2 2 (fun row col -> weights01vector[row][col])
            let biases1 = [|0.2f; -0.1f|]
            let weights02vector = [|[|-0.9f; 0.7f|];[|0.2f;-0.3f|]|]
            let weights02 = Array2D.init 2 2 (fun row col -> weights02vector[row][col])
            let biases2 = [|0.8f;-0.5f|]
            let layer1 = Layer(weights01, biases1, BoundingFunction.ReLu)
            let layer2 = Layer(weights02, biases2, BoundingFunction.Sigmoid)
            NeuralNetwork([|layer1; layer2|])

        testPropNeuralNet.FeedForward([|0.4f;0.6f|])
        |> ignore
        //Steps should show in feed forward (0.4, 0.6) then (0.38, 0.16) then (0.63876, 0.3841429).

        NeuralNetwork.backprop testPropNeuralNet [|0.4f; 0.6f|] [|0.0f; 1.0f|]
        |> ignore
        //Backprop should have a matrix of the final layer being some arrangement of 0.056, 0.02358, -0.05536, and -0.0233116/
        //The next layer should include a delta value os -0.1617915 and a derivative of -0.0697166

        let neuralNet = NeuralNetwork.createRandomizedNeuralNetwork initialLayerHeight [|30; 10|] //This is made from the images

        //Performing some tests
        let newMatrix =
            let arrays1 = [| [|1.0f; 2.0f|]; [|3.0f; 4.0f|]; [|10.0f; 100.0f|] |]
            let arrays2 = [| [|5.0f; 6.0f|]; [|7.0f; 8.0f|] |]
            let matrix1 = Array2D.init 3 2 (fun i j -> arrays1[i][j])
            let matrix2 = Array2D.init 2 2 (fun i j -> arrays2[i][j])
            let matrix = matrix1 %* matrix2
            printfn "%A" matrix
            matrix

        let testNeuralNet = NeuralNetwork.createRandomizedNeuralNetwork 3 [|2; 3; 4|]
        let testInput = [|1.0f; 2.0f; 3.0f|]
        let runResult = snd (testNeuralNet.FeedForward testInput)

        //Back to image net
        let datafiedImages = goodImages |> Array.map DigitImage.ConvertDigitImageToFloats
        let boundTo1DatafiedImages =
            let (inputImages, expectedOutputs) =
                datafiedImages
                |> Array.unzip
            let boundInputImages = inputImages |> Array.map (Array.map (fun x -> x / 255.0f))
            (boundInputImages, expectedOutputs) ||> Array.zip
        NeuralNetwork.SGD boundTo1DatafiedImages 30 20 1.0f TestMode.Test neuralNet |> ignore

        ignore ()

        0