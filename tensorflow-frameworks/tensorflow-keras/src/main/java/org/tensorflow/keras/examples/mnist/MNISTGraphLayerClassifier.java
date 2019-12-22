//package org.tensorflow.keras.examples.mnist;
//
//import org.tensorflow.*;
//import org.tensorflow.data.Dataset;
//import org.tensorflow.data.Pair;
//import org.tensorflow.keras.datasets.MNIST;
//import org.tensorflow.keras.losses.Loss;
//import org.tensorflow.keras.losses.Losses;
//import org.tensorflow.keras.metrics.Metric;
//import org.tensorflow.keras.metrics.Metrics;
//import org.tensorflow.keras.optimizers.GradientDescentOptimizer;
//import org.tensorflow.keras.optimizers.Optimizer;
//import org.tensorflow.op.Ops;
//import org.tensorflow.op.core.Constant;
//import org.tensorflow.data.Tuple2;
//
//import java.io.IOException;
//import java.util.Arrays;
//import java.util.List;
//
//public class MNISTLayerClassifier {
//    private static final int INPUT_SIZE = 28 * 28;
//
//    private static final float LEARNING_RATE = 0.15f;
//    private static final int FEATURES = 10;
//    private static final int BATCH_SIZE = 100;
//
//    private static final int EPOCHS = 10;
//
//    public static void main(String[] args) throws Exception {
//        MNISTLayerClassifier mnist = new MNISTLayerClassifier();
//    }
//
//    public static <T> Operand<T> assume(Operand<T> captured, Class<T> dtype) {
//        return captured;
//    }
//
//    public void trainGraphMode() throws Exception {
//        try (Graph graph = new Graph()) {
//            Ops tf = Ops.create(graph);
//
//            // Load MNIST Dataset
//            Tuple2<Pair<float[][][], float[][]>> data;
//            try {
//                data = MNIST.loadData();
//            } catch (IOException e) {
//                throw new IllegalArgumentException("Could not load MNIST dataset.");
//            }
//
//            List<Operand<?>> trainTensors = Arrays.asList(
//                    Constant.create(tf.scope(), data.first().first()),
//                    Constant.create(tf.scope(), data.first().second())
//            );
//
//            List<Operand<?>> testTensors = Arrays.asList(
//                    Constant.create(tf.scope(), data.second().first()),
//                    Constant.create(tf.scope(), data.second().second())
//            );
//
//            Class<Float> dtype = Float.class;
//
//            Dataset train = Dataset.fromTensorSlices(tf, trainTensors, Arrays.asList(dtype, dtype)).batch(BATCH_SIZE);
//            Dataset test  = Dataset.fromTensorSlices(tf, testTensors,  Arrays.asList(dtype, dtype)).batch(BATCH_SIZE);
//
//            Input<Float> inputLayer = new Input<>(INPUT_SIZE);
//            Dense<Float> denseLayer = new Dense<>(FEATURES,
//                    Dense.Options.builder()
//                            .setActivation(Activations.softmax)
//                            .build());
//
//            Loss loss = Losses.select(Losses.sparseCategoricalCrossentropy);
//            Metric accuracy = Metrics.select(Metrics.accuracy);
//            Optimizer<Float> optimizer = new GradientDescentOptimizer<>(LEARNING_RATE);
//
//            // Compile Model
//            inputLayer.build(tf, dtype);
//            denseLayer.build(tf, inputLayer.computeOutputShape(), dtype);
//            optimizer.build(tf, dtype);
//
//
//            try (Session session = new Session(graph)) {
//                {
//                    // Fit Model (TRAIN)
//                    Pair<Operation, List<Output<?>>> oneShotComponents = train.makeOneShotIterator();
//
//                    Operation makeIterator = oneShotComponents.first();
//
//
//                    Operand<?> XBatch = oneShotComponents.second().get(0);
//                    Operand<?> yBatch = oneShotComponents.second().get(1);
//
//                    // Compute Output / Loss / Accuracy
//                    Operand<Float> yTrue = yBatch;
//                    Operand<Float> yPred = denseLayer.apply(tf, XBatch);
//
//                    Operand<Float> batchLoss = loss.apply(tf, Float.class,  yTrue, yPred);
//                    Operand<Float> batchAccuracy = accuracy.apply(tf, Float.class, yTrue, yPred);
//
//                    List<Operand<Float>> minimize = optimizer.minimize(tf, batchLoss, denseLayer.trainableWeights());
//
//
//
//
//                    // Initialization
//                    session.runner()
//                            .addTargets(denseLayer.initializerOps())
//                            .addTarget(oneShotComponents.first().op())
//                            .run();
//
//                    for (int epoch = 0; epoch < EPOCHS; epoch++) {
//                        float trainEpochAccuracy = 0;
//                        float trainEpochLoss = 0;
//
//                        // Load Batches
//
//                        while (true) {
//
//                        }
//
//                        for (int i = 0; i < train.numBatches(); i++) {
//                            runner = session.runner();
//                            train.feedBatchToSessionRunner(tf, runner, i, false);
//
//                            for (Operand<Float> op : minimize) {
//                                runner.addTarget(op);
//                            }
//
//                            runner.fetch(batchLoss);
//                            runner.fetch(batchAccuracy);
//
//                            List<Tensor<?>> values = runner.run();
//                            try (Tensor<?> lossTensor = values.get(0);
//                                 Tensor<?> accuracyTensor = values.get(1)) {
//                                trainEpochAccuracy += accuracyTensor.floatValue() / train.numBatches();
//                                trainEpochLoss += lossTensor.floatValue() / train.numBatches();
//                            }
//                        }
//
//                        System.out.println("Epoch " + epoch + " train accuracy: " + trainEpochAccuracy + "  loss: " + trainEpochLoss);
//                    }
//                }
//
//
//                {
//                    // Fit Model (TEST)
//                    Session.Runner runner = session.runner();
//                    Operand<Float> XOp = testOps[0];
//                    Operand<Float> yOp = testOps[1];
//
//                    // Compute Output / Loss / Accuracy
//                    Operand<Float> yTrue = yOp;
//                    Operand<Float> yPred = denseLayer.apply(tf, XOp);
//
//                    Operand<Float> batchLoss = loss.apply(tf, Float.class, yTrue, yPred);
//                    Operand<Float> batchAccuracy = accuracy.apply(tf, Float.class, yTrue, yPred);
//
//                    float trainEpochAccuracy = 0;
//                    float trainEpochLoss = 0;
//
//                    // Load Batches
//                    for (int i = 0; i < test.numBatches(); i++) {
//                        runner = session.runner();
//                        test.feedBatchToSessionRunner(tf, runner, i, false);
//                        runner.fetch(batchLoss);
//                        runner.fetch(batchAccuracy);
//
//                        List<Tensor<?>> values = runner.run();
//                        try (Tensor<?> lossTensor = values.get(0);
//                             Tensor<?> accuracyTensor = values.get(1)) {
//                            trainEpochAccuracy += accuracyTensor.floatValue() / test.numBatches();
//                            trainEpochLoss += lossTensor.floatValue() / test.numBatches();
//                        }
//                    }
//
//                    System.out.println("Test accuracy: " + trainEpochAccuracy + "  loss: " + trainEpochLoss);
//                }
//            }
//        }
//    }
//}