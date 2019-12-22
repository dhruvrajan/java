package org.tensorflow.keras.datasets;

import org.tensorflow.data.Pair;
import org.tensorflow.keras.utils.DataUtils;
import org.tensorflow.keras.utils.Keras;
import org.tensorflow.data.Tuple2;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.zip.GZIPInputStream;

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
    public static void download() throws IOException, NoSuchAlgorithmException {
        DataUtils.getFile(LOCAL_PREFIX + TRAIN_IMAGES, ORIGIN_BASE + TRAIN_IMAGES,
                "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609", "SHA-256");
        DataUtils.getFile(LOCAL_PREFIX + TRAIN_LABELS, ORIGIN_BASE + TRAIN_LABELS,
                "fcdfeedb53b53c99384b2cd314206a08fdf6aa97070e19921427a179ea123d19", "SHA-256");
        DataUtils.getFile(LOCAL_PREFIX + TEST_IMAGES, ORIGIN_BASE + TEST_IMAGES,
                "beb4b4806386107117295b2e3e08b4c16a6dfb4f001bfeb97bf25425ba1e08e4","SHA-256");
        DataUtils.getFile(LOCAL_PREFIX + TEST_LABELS, ORIGIN_BASE + TEST_LABELS,
                "986c5b8cbc6074861436f5581f7798be35c7c0025262d33b4df4c9ef668ec773", "SHA-256");
    }


    public static Tuple2<Pair<float[][][], float[][]>> loadData() throws IOException, NoSuchAlgorithmException {
        // Download MNIST files if they don't exist.
        MNIST.download();

        // Read data files into arrays
        float[][][] trainImages = readImages2D(Keras.path(LOCAL_PREFIX, TRAIN_IMAGES).toString());
        float[][] trainLabels = readLabelsOneHot(Keras.path(LOCAL_PREFIX, TRAIN_LABELS).toString());
        float[][][] testImages = readImages2D(Keras.path(LOCAL_PREFIX, TEST_IMAGES).toString());
        float[][] testLabels = readLabelsOneHot(Keras.path(LOCAL_PREFIX + TEST_LABELS).toString());

        // Return a pair of graph loaders; train and test sets
        return new Tuple2<>(Pair.of(trainImages, trainLabels), Pair.of(testImages, testLabels));
    }


    static float[][][] readImages2D(String imagesPath) throws IOException {
        try (DataInputStream inputStream =
                     new DataInputStream(new GZIPInputStream(new FileInputStream(imagesPath)))) {

            if (inputStream.readInt() != IMAGE_MAGIC) {
                throw new IllegalArgumentException("Invalid Image Data File");
            }

            int numImages = inputStream.readInt();
            int rows = inputStream.readInt();
            int cols = inputStream.readInt();

            return readImageBuffer2D(inputStream, numImages, rows, cols);
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
        try (DataInputStream inputStream =
                     new DataInputStream(new GZIPInputStream(new FileInputStream(labelsPath)))) {
            if (inputStream.readInt() != LABELS_MAGIC) {
                throw new IllegalArgumentException("Invalid Label Data File");
            }

            int numLabels = inputStream.readInt();
            return readLabelBuffer(inputStream, numLabels);
        }
    }

    private static byte[][] readBatchedBytes(DataInputStream inputStream, int batches, int bytesPerBatch) throws IOException {
        byte[][] entries = new byte[batches][bytesPerBatch];
        for (int i = 0; i < batches; i++) {
            inputStream.readFully(entries[i]);
        }
        return entries;
    }

    private static float[][][] readImageBuffer2D(
            DataInputStream inputStream, int numImages, int imageWidth, int imageHeight) throws IOException {
        float[][][] unsignedEntries = new float[numImages][imageWidth][imageHeight];
        for (int i = 0; i < unsignedEntries.length; i++) {
            byte[][] entries = readBatchedBytes(inputStream, imageWidth, imageHeight);
            for (int j = 0; j < unsignedEntries[0].length; j++) {
                for (int k = 0; k < unsignedEntries[0][0].length; k++) {
                    unsignedEntries[i][j][k] = (float) (entries[j][k] & 0xFF) / 255.0f;
                }
            }
        }
        return unsignedEntries;
    }

    private static float[][] readLabelBuffer(DataInputStream inputStream, int numLabels)
            throws IOException {
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

        if (fill) Arrays.fill(oneHot, 0);
        oneHot[label] = 1.0f;
    }
}
