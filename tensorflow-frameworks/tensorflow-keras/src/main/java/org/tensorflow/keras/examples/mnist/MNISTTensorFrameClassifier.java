package org.tensorflow.keras.examples.mnist;

import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.data.GraphTensorArrayDataset;
import org.tensorflow.keras.activations.Activations;
import org.tensorflow.keras.datasets.MNIST;
import org.tensorflow.keras.layers.Dense;
import org.tensorflow.keras.layers.Input;
import org.tensorflow.keras.losses.Loss;
import org.tensorflow.keras.losses.Losses;
import org.tensorflow.keras.metrics.Metric;
import org.tensorflow.keras.metrics.Metrics;
import org.tensorflow.keras.optimizers.GradientDescentOptimizer;
import org.tensorflow.keras.optimizers.Optimizer;
import org.tensorflow.op.Ops;
import org.tensorflow.utils.Pair;

import java.io.IOException;
import java.util.List;

public class MNISTTensorFrameClassifier  {
    private static final int INPUT_SIZE = 28 * 28;

    private static final float LEARNING_RATE = 0.15f;
    private static final int FEATURES = 10;
    private static final int BATCH_SIZE = 100;

    private static final int EPOCHS = 10;

    public static void main(String[] args) throws Exception {
        MNISTTensorFrameClassifier mnist = new MNISTTensorFrameClassifier();
        mnist.run();

    }

    public void run() throws Exception {
        try (Graph graph = new Graph()) {
            Ops tf = Ops.create(graph);

            // Load MNIST Dataset
            Pair<GraphTensorArrayDataset<Float>, GraphTensorArrayDataset<Float>> data;
            try {
                data = MNIST.graphLoaders(graph);
            } catch (IOException e) {
                throw new IllegalArgumentException("Could not load MNIST dataset.");
            }

            try (GraphTensorArrayDataset<Float> train = data.first();
                 GraphTensorArrayDataset<Float> test = data.second()) {

                Input<Float> inputLayer = new Input<>(INPUT_SIZE);
                Dense<Float> denseLayer = new Dense<>(FEATURES, Dense.Options.builder().setActivation(Activations.softmax).build());

                Loss loss = Losses.select(Losses.sparseCategoricalCrossentropy);
                Metric accuracy = Metrics.select(Metrics.accuracy);
                Optimizer<Float> optimizer = new GradientDescentOptimizer<>(LEARNING_RATE);

                // Compile Model
                train.batch(BATCH_SIZE);
                train.build(tf);
                Operand<Float>[] trainOps = train.getBatchOperands();

                test.batch(BATCH_SIZE);
                test.build(tf);
                Operand<Float>[] testOps = test.getBatchOperands();

                inputLayer.build(tf, Float.class);
                denseLayer.build(tf, inputLayer.computeOutputShape(), Float.class);
                optimizer.build(tf, Float.class);


                // Fit Model (TRAIN)
                try (Session session = new Session(graph)) {
                    {
                        Session.Runner runner = session.runner();
                        Operand<Float> XOp = trainOps[0];
                        Operand<Float> yOp = trainOps[1];

                        // Compute Output / Loss / Accuracy
                        Operand<Float> yTrue = yOp;
                        Operand<Float> yPred = denseLayer.apply(tf, XOp);

                        Operand<Float> batchLoss = loss.apply(tf, Float.class,  yTrue, yPred);
                        Operand<Float> batchAccuracy = accuracy.apply(tf, Float.class, yTrue, yPred);

                        List<Operand<Float>> minimize = optimizer.minimize(tf, batchLoss, denseLayer.trainableWeights());


                        // Run initializer ops
                        for (Operand<Float> op : denseLayer.initializerOps()) {

                            runner.addTarget(op);
                        }

                        runner.run();

                        for (int epoch = 0; epoch < EPOCHS; epoch++) {
                            float trainEpochAccuracy = 0;
                            float trainEpochLoss = 0;

                            // Load Batches
                            for (int i = 0; i < train.numBatches(); i++) {
                                runner = session.runner();
                                train.feedBatchToSessionRunner(tf, runner, i, false);

                                for (Operand<Float> op : minimize) {
                                    runner.addTarget(op);
                                }

                                runner.fetch(batchLoss);
                                runner.fetch(batchAccuracy);

                                List<Tensor<?>> values = runner.run();
                                try (Tensor<?> lossTensor = values.get(0);
                                     Tensor<?> accuracyTensor = values.get(1)) {
                                    trainEpochAccuracy += accuracyTensor.floatValue() / train.numBatches();
                                    trainEpochLoss += lossTensor.floatValue() / train.numBatches();
                                }
                            }

                            System.out.println("Epoch " + epoch + " train accuracy: " + trainEpochAccuracy + "  loss: " + trainEpochLoss);
                        }
                    }

                    // Fit Model (TEST)
                    {
                        Session.Runner runner = session.runner();
                        Operand<Float> XOp = testOps[0];
                        Operand<Float> yOp = testOps[1];

                        // Compute Output / Loss / Accuracy
                        Operand<Float> yTrue = yOp;
                        Operand<Float> yPred = denseLayer.apply(tf, XOp);

                        Operand<Float> batchLoss = loss.apply(tf, Float.class, yTrue, yPred);
                        Operand<Float> batchAccuracy = accuracy.apply(tf, Float.class, yTrue, yPred);

                        float trainEpochAccuracy = 0;
                        float trainEpochLoss = 0;

                        // Load Batches
                        for (int i = 0; i < test.numBatches(); i++) {
                            runner = session.runner();
                            test.feedBatchToSessionRunner(tf, runner, i, false);
                            runner.fetch(batchLoss);
                            runner.fetch(batchAccuracy);

                            List<Tensor<?>> values = runner.run();
                            try (Tensor<?> lossTensor = values.get(0);
                                 Tensor<?> accuracyTensor = values.get(1)) {
                                trainEpochAccuracy += accuracyTensor.floatValue() / test.numBatches();
                                trainEpochLoss += lossTensor.floatValue() / test.numBatches();
                            }
                        }

                        System.out.println("Test accuracy: " + trainEpochAccuracy + "  loss: " + trainEpochLoss);
                    }
                }
            }
        }
    }
}