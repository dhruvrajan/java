package org.tensorflow.keras.examples.mnist;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.data.Dataset;
import org.tensorflow.data.OneShotIterator;
import org.tensorflow.keras.datasets.MNIST;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Assign;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Gradients;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.train.ApplyGradientDescent;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt64;
import org.tensorflow.utils.Pair;

/**
 * An example showing a simple feed-forward classifier for MNIST
 * using tf.data and core TensorFlow (in Graph Mode).
 */
public class MNISTBasicGraphClassifier implements Runnable {
    private static final int INPUT_SIZE = 28 * 28;

    private static final float LEARNING_RATE = 0.2f;
    private static final int FEATURES = 10;
    private static final int BATCH_SIZE = 100;
    private static final int EPOCHS = 10;

    public static void main(String[] args) {
        new MNISTBasicGraphClassifier().run();
    }

    public Operand<TFloat32> predict(Ops tf, Operand<TFloat32> images, Variable<TFloat32> weights,
            Variable<TFloat32> biases) {
        return tf.nn.softmax(tf.math.add(tf.linalg.matMul(images, weights), biases));
    }

    public Operand<TFloat32> crossEntropyLoss(Ops tf, Operand<TFloat32> predicted, Operand<TFloat32> actual) {
        return tf.math.mean(tf.math.neg(tf.reduceSum(tf.math.mul(actual, tf.math.log(predicted)), tf.constant(1))),
                tf.constant(0));
    }

    public List<ApplyGradientDescent<TFloat32>> optimizerTargetOps(Ops tf, Operand<TFloat32> loss,
            Operand<TFloat32> weights, Operand<TFloat32> biases) {
        Gradients gradients = tf.gradients(loss, Arrays.asList(weights, biases));
        Constant<TFloat32> alpha = tf.constant(LEARNING_RATE);

        ApplyGradientDescent<TFloat32> weightGradientDescent = tf.train.applyGradientDescent(weights, alpha,
                gradients.dy(0));
        ApplyGradientDescent<TFloat32> biasGradientDescent = tf.train.applyGradientDescent(biases, alpha,
                gradients.dy(1));

        return Arrays.asList(weightGradientDescent, biasGradientDescent);
    }

    public Operand<TFloat32> accuracy(Ops tf, Operand<TFloat32> predicted, Operand<TFloat32> actual) {
        Operand<TInt64> yHat = tf.math.argMax(predicted, tf.constant(1));
        Operand<TInt64> yTrue = tf.math.argMax(actual, tf.constant(1));
        Operand<TFloat32> accuracy = tf.math.mean(tf.dtypes.cast(tf.math.equal(yHat, yTrue), TFloat32.DTYPE),
                tf.constant(0));
        return accuracy;
    }


    public void run() {
        try (Graph graph = new Graph()) {
            Ops tf = Ops.create(graph);

            // Load datasets
            Pair<Dataset> trainTestSplit;
            try {
                trainTestSplit = MNIST.loadDatasets(tf);
            } catch (IOException e) {
                System.out.println("Could not load dataset.");
                return;
            }
            Dataset trainDataset = trainTestSplit.first().batch(BATCH_SIZE);
            Dataset testDataset = trainTestSplit.second().batch(BATCH_SIZE);

            // Extract iterators and train / test components
            OneShotIterator trainIterator = trainDataset.makeOneShotIterator();
            List<Output<?>> components = trainIterator.getComponents();
            Operand<TFloat32> trainImages = components.get(0).expect(TFloat32.DTYPE);
            Operand<TFloat32> trainLabels = components.get(1).expect(TFloat32.DTYPE);

            OneShotIterator testIterator = testDataset.makeOneShotIterator();
            List<Output<?>> testComponents = testIterator.getComponents();
            Operand<TFloat32> testImages = testComponents.get(0).expect(TFloat32.DTYPE);
            Operand<TFloat32> testLabels = testComponents.get(1).expect(TFloat32.DTYPE);

            // Flatten image tensors
            trainImages = tf.reshape(trainImages, tf.constant(new int[] { -1, INPUT_SIZE }));
            testImages = tf.reshape(testImages, tf.constant(new int[] { -1, INPUT_SIZE }));

            // Declare, initialize weights
            Variable<TFloat32> weights = tf.variable(Shape.make(INPUT_SIZE, FEATURES), TFloat32.DTYPE);
            Assign<TFloat32> weightsInit = tf.assign(weights,
                    tf.zeros(tf.constant(new int[] { INPUT_SIZE, FEATURES }), TFloat32.DTYPE));

            Variable<TFloat32> biases = tf.variable(Shape.make(FEATURES), TFloat32.DTYPE);
            Assign<TFloat32> biasesInit = tf.assign(biases,
                    tf.zeros(tf.constant(new int[] { FEATURES }), TFloat32.DTYPE));

            // SETUP: Training
            Operand<TFloat32> trainPrediction = predict(tf, trainImages, weights, biases);
            Operand<TFloat32> trainAccuracy = accuracy(tf, trainPrediction, trainLabels);

            Operand<TFloat32> trainLoss = crossEntropyLoss(tf, trainPrediction, trainLabels);
            List<ApplyGradientDescent<TFloat32>> optimizerTargets = optimizerTargetOps(tf, trainLoss, weights, biases);

            // SETUP: Testing
            Operand<TFloat32> testPrediction = predict(tf, testImages, weights, biases);
            Operand<TFloat32> testAccuracy = accuracy(tf, testPrediction, testLabels);

            try (Session session = new Session(graph)) {

                // Initialize weights and biases
                session.runner().addTarget(weightsInit).addTarget(biasesInit).run();

                // Run training loop
                for (int i = 0; i < EPOCHS; i++) {
                    // reset iterator object
                    session.runner().addTarget(trainIterator.getMakeIteratorOp()).run();

                    int batches = 0;
                    float epochAccuracy = 0;
                    while (true) {
                        try {
                            List<Tensor<?>> fetched = session.runner()
                                .addTarget(optimizerTargets.get(0))
                                .addTarget(optimizerTargets.get(1))
                                .fetch(trainAccuracy)
                                .run();

                            float batchAccuracy = fetched.get(0).floatValue();
                            epochAccuracy += batchAccuracy;
                            batches += 1;
                        } catch (IndexOutOfBoundsException e) {
                            break;
                        }
                    }

                    System.out.println("Epoch Accuracy " + i + ": " + epochAccuracy / batches);
                }

                // Evaluate on test set
                session.runner().addTarget(testIterator.getMakeIteratorOp()).run();

                int batches = 0;
                float epochAccuracy = 0;
                while (true) {
                    try {
                        List<Tensor<?>> fetched = session.runner().fetch(testAccuracy).run();

                        float batchAccuracy = fetched.get(0).floatValue();
                        epochAccuracy += batchAccuracy;
                        batches += 1;
                    } catch (IndexOutOfBoundsException e) {
                        break;
                    }
                }

                System.out.println("Test Accuracy " + ": " + epochAccuracy / batches);

            }
        }
    }
}
