import scala.collection.JavaConversions._

import io.skymind.zeppelin.utils._
import io.skymind.modelproviders.history.client.ModelHistoryClient
import io.skymind.modelproviders.history.model._

import org.deeplearning4j.datasets.iterator._
import org.deeplearning4j.datasets.iterator.impl._
import org.deeplearning4j.nn.api._
import org.deeplearning4j.nn.multilayer._
import org.deeplearning4j.nn.graph._
import org.deeplearning4j.nn.conf._
import org.deeplearning4j.nn.conf.inputs._
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex
import org.deeplearning4j.nn.weights._
import org.deeplearning4j.optimize.listeners._
import org.deeplearning4j.api.storage.impl.RemoteUIStatsStorageRouter
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator
import org.deeplearning4j.eval.Evaluation

import org.datavec.api.transform._
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.SequenceRecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit

import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config._
import org.nd4j.linalg.lossfunctions.LossFunctions._
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.primitives.Pair
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.MultiDataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerStandardize

import java.io.File
import java.net.URL
import java.util.ArrayList
import java.util.Collections
import java.util.List
import java.util.Random

import org.apache.commons.io.IOUtils
import org.apache.commons.io.FileUtils

val skilContext = new SkilContext()
val client = skilContext.client

val baseDir: File = new File("/tmp/uci-data")
val baseTrainDir: File = new File(baseDir, "train")
val featuresDirTrain: File = new File(baseTrainDir, "features")
val labelsDirTrain: File = new File(baseTrainDir, "labels")
val baseTestDir: File = new File(baseDir, "test")
val featuresDirTest: File = new File(baseTestDir, "features")
val labelsDirTest: File = new File(baseTestDir, "labels")

def downloadUCIData() {
    //Data already exists
    if (baseDir.exists()) {
        print (s"directory $baseDir already exists")
        return
    }

    val url: String =
        "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data"
    val data: String = IOUtils.toString(new URL(url))
    val lines: Array[String] = data.split("\n")
    print (s"downloaded data from $url to $baseDir")

    // Perhaps redundant / unreachable code!
    if (baseDir.exists()) {
        baseDir.delete()
    }

    baseDir.mkdir()
    baseTrainDir.mkdir()
    featuresDirTrain.mkdir()
    labelsDirTrain.mkdir()
    baseTestDir.mkdir()
    featuresDirTest.mkdir()
    labelsDirTest.mkdir()

    var lineCount: Int = 0
    val contentAndLabels: List[Pair[String, Integer]] =
        new ArrayList[Pair[String, Integer]]()

    for (line <- lines) {
        val transposed: String = line.replaceAll(" +", "\n")
        //Labels: first 100 are label 0, second 100 are label 1, and so on

        contentAndLabels.add(new Pair(transposed, (lineCount / 100)))
        lineCount += 1
    }

    // Randomize and do a train/test split:
    Collections.shuffle(contentAndLabels, new Random(12345))

    //75% train, 25% test
    val nTrain: Int = 450
    var trainCount: Int = 0
    var testCount: Int = 0
    for (p <- contentAndLabels) {
        var outPathFeatures: File = null
        var outPathLabels: File = null
        if (trainCount < nTrain) {
            outPathFeatures = new File(featuresDirTrain, trainCount + ".csv")
            outPathLabels = new File(labelsDirTrain, trainCount + ".csv") 
            trainCount += 1
        } else {
            outPathFeatures = new File(featuresDirTest, testCount + ".csv")
            outPathLabels = new File(labelsDirTest, testCount + ".csv")
            testCount += 1
        }

        FileUtils.writeStringToFile(outPathFeatures, p.getFirst)
        FileUtils.writeStringToFile(outPathLabels, p.getSecond.toString)
    }
}

// Download data as needed
downloadUCIData()

// Load the training data
val trainFeatures: SequenceRecordReader = new CSVSequenceRecordReader()
trainFeatures.initialize(
    new NumberedFileInputSplit(
        featuresDirTrain.getAbsolutePath + "/%d.csv",
        0,
        449))

val trainLabels: RecordReader = new CSVRecordReader()
trainLabels.initialize(new NumberedFileInputSplit(
    labelsDirTrain.getAbsolutePath + "/%d.csv",
    0,
    449))

val minibatch: Int = 10
val numLabelClasses: Int = 6

val trainData: MultiDataSetIterator = new RecordReaderMultiDataSetIterator.Builder(minibatch)
    .addSequenceReader("features", trainFeatures)
    .addReader("labels", trainLabels)
    .addInput("features")
    .addOutputOneHot("labels", 0, numLabelClasses)
    .build()

// Normalize the training data
def makeNormalizer( mds:MultiDataSetIterator ) : MultiNormalizerStandardize = {
    val n = new MultiNormalizerStandardize()

    // Collect training data statistics
    n.fit(mds)
    mds.reset()
    return n
}

val normalizer = makeNormalizer(trainData)
val mean = normalizer.getFeatureMean(0)
val std = normalizer.getFeatureStd(0)

println(s"Mean: $mean, Std: $std")

// Use previously collected statistics to normalize on-the-fly
trainData.setPreProcessor(normalizer)

// Load the test data
val testFeatures: SequenceRecordReader = new CSVSequenceRecordReader()
testFeatures.initialize(new NumberedFileInputSplit(
    featuresDirTest.getAbsolutePath + "/%d.csv",
    0,
    149))

val testLabels: RecordReader = new CSVRecordReader()
testLabels.initialize(
    new NumberedFileInputSplit(labelsDirTest.getAbsolutePath + "/%d.csv",
    0,
    149))

val testData: MultiDataSetIterator = new RecordReaderMultiDataSetIterator.Builder(minibatch)
    .addSequenceReader("features", testFeatures)
    .addReader("labels", testLabels)
    .addInput("features")
    .addOutputOneHot("labels", 0, numLabelClasses)
    .build()

testData.setPreProcessor(normalizer)

// Configure the network
val conf: ComputationGraphConfiguration = new NeuralNetConfiguration.Builder()
    .seed(123)
    .trainingWorkspaceMode(WorkspaceMode.SINGLE)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .iterations(1)
    .weightInit(WeightInit.XAVIER)
    .updater(new Nesterovs(0.005, 0.9))
    .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
    .gradientNormalizationThreshold(0.5)
    .graphBuilder()
    .addInputs("input")
    .setInputTypes(InputType.recurrent(1))
    .addLayer("lstm", new GravesLSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10).build(), "input")
    .addVertex("pool", new LastTimeStepVertex("input"), "lstm")
    .addLayer("output", new OutputLayer.Builder(LossFunction.MCXENT)
        .activation(Activation.SOFTMAX).nIn(10).nOut(numLabelClasses).build(), "pool")
    .setOutputs("output")
    .pretrain(false)
    .backprop(true)
    .build()

val network_model: ComputationGraph = new ComputationGraph(conf)
network_model.init()

network_model.setListeners(new ScoreIterationListener(20))

// Train the network, evaluating the test set performance at each step
trainData.reset()
testData.reset()

val nEpochs: Int = 10

for (i <- 0 until nEpochs) {
    network_model.fit(trainData)

    // Evaluate on the test set:
    val roc = new ROCMultiClass()
    val eval = new Evaluation()
    network_model.doEvaluation(testData, roc);
    println(roc.stats())

    val evaluation: Evaluation = network_model.evaluate(testData)
    var accuracy = evaluation.accuracy()
    var f1 = evaluation.f1()

    println(s"Test set evaluation at epoch $i: Accuracy = $accuracy, F1 = $f1")

    testData.reset()
    trainData.reset()
}

// Save Model
var evaluation = network_model.evaluate(testData)
val modelId = skilContext.addModelToExperiment(z, network_model)
val evalId = skilContext.addEvaluationToModel(z, modelId, evaluation)
