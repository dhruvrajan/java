package org.tensorflow.keras.datasets;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.zip.GZIPInputStream;

import org.tensorflow.data.Dataset;
import org.tensorflow.keras.utils.DataUtils;
import org.tensorflow.keras.utils.Keras;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TFloat32;
import org.tensorflow.utils.Pair;
import org.tensorflow.utils.Tuple2;

/**
 * Code based on example found at:
 * https://github.com/karllessard/models/tree/master/samples/languages/java/mnist/src/main/java/org/tensorflow/model/sample/mnist
 * <p>
 * Utility for downloading and using MNIST data with a local keras installation.
 */
public class MNIST {
    private static final int IMAGE_MAGIC = 2051;
    private static final int LABELS_MAGIC = 2049;
    private static final int OUTPUT_CLASSES = 10;

    private static final String TRAIN_IMAGES = "train-images-idx3-ubyte.gz";
    private static final String TRAIN_LABELS = "train-labels-idx1-ubyte.gz";
    private static final String TEST_IMAGES = "t10k-images-idx3-ubyte.gz";
    private static final String TEST_LABELS = "t10k-labels-idx1-ubyte.gz";

    private static final String ORIGIN_BASE = "http://yann.lecun.com/exdb/mnist/";

    private static final String LOCAL_PREFIX = "datasets/mnist/";

    /**
     * Download MNIST dataset files to the local .keras/ directory.
     *
     * @throws IOException when the download fails
     */
    public static void download() throws IOException {
        DataUtils.getFile(LOCAL_PREFIX + TRAIN_IMAGES, ORIGIN_BASE + TRAIN_IMAGES);
        DataUtils.getFile(LOCAL_PREFIX + TRAIN_LABELS, ORIGIN_BASE + TRAIN_LABELS);
        DataUtils.getFile(LOCAL_PREFIX + TEST_IMAGES, ORIGIN_BASE + TEST_IMAGES);
        DataUtils.getFile(LOCAL_PREFIX + TEST_LABELS, ORIGIN_BASE + TEST_LABELS);
    }

    public static Pair<Dataset> loadDatasets(Ops tf) throws IOException {
        Pair<Tuple2<float[][][], float[][]>> mnist = loadData();

        Dataset train = Dataset.fromTensorSlices(tf,
                Arrays.asList(tf.constant(mnist.first().first()), tf.constant(mnist.first().second())),
                Arrays.asList(TFloat32.DTYPE, TFloat32.DTYPE));

        Dataset test = Dataset.fromTensorSlices(tf,
                Arrays.asList(tf.constant(mnist.second().first()), tf.constant(mnist.second().second())),
                Arrays.asList(TFloat32.DTYPE, TFloat32.DTYPE));

        return new Pair<>(train, test);
    }

    public static Pair<Tuple2<float[][][], float[][]>> loadData() throws IOException {
        // Download MNIST files if they don't exist.
        MNIST.download();

        // Construct local data file paths
        String trainImagesPath = Keras.path(LOCAL_PREFIX, TRAIN_IMAGES).toString();
        String trainLabelsPath = Keras.path(LOCAL_PREFIX, TRAIN_LABELS).toString();

        String testImagesPath = Keras.path(LOCAL_PREFIX, TEST_IMAGES).toString();
        String testLabelsPath = Keras.path(LOCAL_PREFIX, TEST_LABELS).toString();

        // Read data files into arrays
        float[][][] trainImages = readImages2D(trainImagesPath);
        float[][] trainLabels = readLabelsOneHot(trainLabelsPath);

        float[][][] testImages = readImages2D(testImagesPath);
        float[][] testLabels = readLabelsOneHot(testLabelsPath);

        // Return a pair of graph loaders; train and test sets
        return new Pair<>(new Tuple2<>(trainImages, trainLabels), new Tuple2<>(testImages, testLabels));
    }

    /**
     * Reads MNIST image files into an array, given an image datafile path.
     *
     * @param labelsPath MNIST image datafile path
     * @return an array of shape (# examples, # classes) containing the label data
     * @throws IOException when the file reading fails.
     */
    static float[][][] readImages2D(String imagesPath) throws IOException {
        try (DataInputStream inputStream = new DataInputStream(new GZIPInputStream(new FileInputStream(imagesPath)))) {

            if (inputStream.readInt() != IMAGE_MAGIC) {
                throw new IllegalArgumentException("Invalid Image Data File");
            }

            int numImages = inputStream.readInt();
            int rows = inputStream.readInt();

            return readImageBuffer2D(inputStream, numImages, rows);
        }
    }

    /**
     * Reads MNIST label files into an array, given a label datafile path.
     *
     * @param labelsPath MNIST label datafile path
     * @return an array of shape (# examples, # classes) containing the label data
     * @throws IOException when the file reading fails.
     */
    static float[][] readLabelsOneHot(String labelsPath) throws IOException {
        try (DataInputStream inputStream = new DataInputStream(new GZIPInputStream(new FileInputStream(labelsPath)))) {
            if (inputStream.readInt() != LABELS_MAGIC) {
                throw new IllegalArgumentException("Invalid Label Data File");
            }

            int numLabels = inputStream.readInt();
            return readLabelBuffer(inputStream, numLabels);
        }
    }

    public static void readLabelBytesToFloatBuffer(DataInputStream inputStream, FloatBuffer fb, int numBytes)
            throws IOException {
        // Read Bytes
        byte[] bytes = new byte[numBytes];
        inputStream.readFully(bytes);

        // Convert Bytes to Float Labels
        float[] floats = new float[numBytes * OUTPUT_CLASSES];
        for (int i = 0; i < numBytes; i++) {
            int label = bytes[i] & 0xFF;
            float[] labelOneHot = labelToOneHotVector(label, true);
            for (int j = 0; j < labelOneHot.length; j++) {
                floats[i + j] = labelOneHot[j];
            }
        }
        // Write floats to buffer
        fb.put(floats);
    }

    public static void readImageBytesToFloatBuffer(DataInputStream inputStream, FloatBuffer fb, int numBytes)
            throws IOException {
        // Read Bytes
        byte[] bytes = new byte[numBytes];
        inputStream.readFully(bytes);

        // Convert Bytes to Floats
        float[] floats = new float[numBytes];
        for (int i = 0; i < numBytes; i++) {
            floats[i] = bytes[i] & 0xFF;
        }

        // Write floats to buffer
        fb.put(floats);
    }

    private static byte[][] readBatchedBytes(DataInputStream inputStream, int batches, int bytesPerBatch)
            throws IOException {
        byte[][] entries = new byte[batches][bytesPerBatch];
        for (int i = 0; i < batches; i++) {
            inputStream.readFully(entries[i]);
        }
        return entries;
    }

    private static float[][][] readImageBuffer2D(DataInputStream inputStream, int numImages, int imageWidth)
            throws IOException {
        float[][][] unsignedEntries = new float[numImages][imageWidth][imageWidth];
        for (int i = 0; i < unsignedEntries.length; i++) {
            byte[][] entries = readBatchedBytes(inputStream, imageWidth, imageWidth);
            for (int j = 0; j < unsignedEntries[0].length; j++) {
                for (int k = 0; k < unsignedEntries[0][0].length; k++) {
                    unsignedEntries[i][j][k] = (float) (entries[j][k] & 0xFF) / 255.0f;
                }
            }
        }
        return unsignedEntries;
    }

    private static float[][] readLabelBuffer(DataInputStream inputStream, int numLabels) throws IOException {
        byte[][] entries = readBatchedBytes(inputStream, numLabels, 1);

        float[][] labels = new float[numLabels][OUTPUT_CLASSES];
        for (int i = 0; i < entries.length; i++) {
            labelToOneHotVector(entries[i][0] & 0xFF, labels[i], false);
        }

        return labels;
    }

    private static void labelToOneHotVector(int label, float[] oneHot, boolean fill) {
        if (label >= oneHot.length) {
            throw new IllegalArgumentException("Invalid Index for One-Hot Vector");
        }

        if (fill)
            Arrays.fill(oneHot, 0);
        oneHot[label] = 1.0f;
    }

    private static float[] labelToOneHotVector(int label, boolean fill) {
        float[] oneHot = new float[OUTPUT_CLASSES];
        if (label >= oneHot.length) {
            throw new IllegalArgumentException("Invalid Index for One-Hot Vector");
        }

        if (fill)
            Arrays.fill(oneHot, 0);
        oneHot[label] = 1.0f;
        return oneHot;
    }
}
