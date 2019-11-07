package org.tensorflow.keras.utils;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

public class Keras {

    // Manage .keras configuration location
    private static final String DEFAULT_KERAS_HOME
            = Paths.get(System.getProperty("user.home"), ".keras").toString();
    private static final String SYSTEM_KERAS_HOME_VAR = "KERAS_HOME";


    public static String home() {
        String systemHome = System.getenv(SYSTEM_KERAS_HOME_VAR);
        return systemHome != null ? systemHome : DEFAULT_KERAS_HOME;
    }

    public static Path path(String... path) {
        return Paths.get(home(), path);
    }

    public static String datasetsDirectory() {
        return Paths.get(home(), "datasets").toString();
    }

    //
    // Keras backend utilties
    //
    public static Operand<Integer> constArray(Ops tf, int... i) {
        return Constant.create(tf.scope(), i);
    }

    public static Operand<Integer> shapeOperand(Ops tf, Shape shape) {
        int[] shapeArray = new int[shape.numDimensions()];
        for (int i = 0; i < shapeArray.length; i++) {
            shapeArray[i] = (int) shape.size(i);
        }

        return Constant.create(tf.scope(), shapeArray);
    }

    public static long head(long... dims) {
        return dims[0];
    }

    public static long[] tail(long... dims) {
        return Arrays.copyOfRange(dims, 1, dims.length);
    }

    public static long[] concatenate(long[] first, long last) {
        long[] dims = new long[first.length + 1];
        System.arraycopy(first, 0, dims, 0, first.length);
        dims[dims.length - 1] = last;
        return dims;
    }

    public static long[] concatenate(long first, long... remaining) {
        long[] dims = new long[remaining.length + 1];
        System.arraycopy(remaining, 0, dims, 1, remaining.length);
        dims[0] = first;
        return dims;
    }

    public static long[] dimsFromShape(Shape shape) {
        long[] dims = new long[shape.numDimensions()];
        for (int i = 0; i < shape.numDimensions(); i++) {
            dims[i] = shape.size(i);
        }
        return dims;
    }

    public static void printFloatTensor(Tensor<?> tensor) {
        FloatBuffer buffer = FloatBuffer.allocate(new TensorShape(shapeFromDims(tensor.shape())).numElements());
        tensor.writeTo(buffer);
        System.out.println(Arrays.toString(buffer.array()));
    }

    public static void printIntTensor(Tensor<?> tensor) {
        IntBuffer buffer = IntBuffer.allocate(new TensorShape(shapeFromDims(tensor.shape())).numElements());
        tensor.writeTo(buffer);
        System.out.println(Arrays.toString(buffer.array()));
    }

    public static void printBoolTensor(Tensor<?> tensor) {
        ByteBuffer buffer = ByteBuffer.allocate(new TensorShape(shapeFromDims(tensor.shape())).numElements());
        tensor.writeTo(buffer);
        System.out.println(Arrays.toString(buffer.array()));
    }

    public static Shape shapeFromDims(long... dims) {
        return Shape.make(head(dims), tail(dims));
    }
}
