package org.tensorflow.keras.examples.mnist;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.tensorflow.EagerSession;
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
 * An example showing a simple feed-forward classifier for MNIST using tf.data
 * and core TensorFlow (in Eager Mode).
 */
public class MNISTBasicEagerClassifier implements Runnable {
    private static final int INPUT_SIZE = 28 * 28;

    private static final float LEARNING_RATE = 0.2f;
    private static final int FEATURES = 10;
    private static final int BATCH_SIZE = 100;
    private static final int EPOCHS = 10;

    public static void main(String[] args) {
        new MNISTBasicEagerClassifier().run();
    }

    public Operand<TFloat32> predict(Ops tf, Operand<TFloat32> images, Variable<TFloat32> weights,
            Variable<TFloat32> biases) {
        return tf.nn.softmax(tf.math.add(tf.linalg.matMul(images, weights), biases));
    }

    public Operand<TFloat32> crossEntropyLoss(Ops tf, Operand<TFloat32> predicted, Operand<TFloat32> actual) {
        return tf.math.mean(tf.math.neg(tf.reduceSum(tf.math.mul(actual, tf.math.log(predicted)), tf.constant(1))),
                tf.constant(0));
    }

    public Operand<TFloat32> accuracy(Ops tf, Operand<TFloat32> predicted, Operand<TFloat32> actual) {
        Operand<TInt64> yHat = tf.math.argMax(predicted, tf.constant(1));
        Operand<TInt64> yTrue = tf.math.argMax(actual, tf.constant(1));
        Operand<TFloat32> accuracy = tf.math.mean(tf.dtypes.cast(tf.math.equal(yHat, yTrue), TFloat32.DTYPE),
                tf.constant(0));
        return accuracy;
    }

    public void run() {
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);

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

            // Declare, initialize weights and constants
            Variable<TFloat32> weights = tf.variable(Shape.make(INPUT_SIZE, FEATURES), TFloat32.DTYPE);
            Variable<TFloat32> biases = tf.variable(Shape.make(FEATURES), TFloat32.DTYPE);
            Constant<TFloat32> alpha = tf.constant(LEARNING_RATE);

            // Initialize model weights
            tf.assign(weights, tf.zeros(tf.constant(new int[] { INPUT_SIZE, FEATURES }), TFloat32.DTYPE));
            tf.assign(biases, tf.zeros(tf.constant(new int[] { FEATURES }), TFloat32.DTYPE));

            // Run training loop
            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                int numBatches = 0;
                float epochAccuracy = 0;

                // Iterate through all batches in dataset
                for (List<Output<?>> trainBatch : trainDataset) {
                    // Extract images and labels
                    Operand<TFloat32> trainImages = trainBatch.get(0).expect(TFloat32.DTYPE);
                    Operand<TFloat32> trainLabels = trainBatch.get(1).expect(TFloat32.DTYPE);

                    // Flatten image tensors
                    trainImages = tf.reshape(trainImages, tf.constant(new int[] { -1, INPUT_SIZE }));

                    Operand<TFloat32> trainPrediction = predict(tf, trainImages, weights, biases);
                    Operand<TFloat32> trainLoss = crossEntropyLoss(tf, trainPrediction, trainLabels);

                    // Calculate gradients
                    Gradients gradients = tf.gradients(trainLoss, Arrays.asList(weights, biases));

                    tf.train.applyGradientDescent(weights, alpha, gradients.dy(0));
                    tf.train.applyGradientDescent(biases, alpha, gradients.dy(1));

                    Operand<TFloat32> trainBatchAccuracy = accuracy(tf, trainPrediction, trainLabels);

                    epochAccuracy += trainBatchAccuracy.asOutput().tensor().floatValue();
                }

                System.out.println("Epoch #" + epoch + ": accuracy=" + epochAccuracy / numBatches);
            }

            // Evaluate on test set
            float numTestBatches = 0;
            float testAccuracy = 0;
            for (List<Output<?>> testBatch : testDataset) {
                // Extract images and labels
                Operand<TFloat32> testImages = testBatch.get(0).expect(TFloat32.DTYPE);
                Operand<TFloat32> testLabels = testBatch.get(1).expect(TFloat32.DTYPE);

                // Flatten image tensors
                testImages = tf.reshape(testImages, tf.constant(new int[] { -1, INPUT_SIZE }));

                Operand<TFloat32> testPrediction = predict(tf, testImages, weights, biases);
                Operand<TFloat32> testBatchAccuracy = accuracy(tf, testPrediction, testLabels);

                testAccuracy += testBatchAccuracy.asOutput().tensor().floatValue();
            }

            System.out.println("Test accuracy=" + testAccuracy / numTestBatches);
        }
    }
}
