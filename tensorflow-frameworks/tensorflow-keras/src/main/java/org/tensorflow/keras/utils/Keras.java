package org.tensorflow.keras.utils;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

public class Keras {
    private static final String KERAS_DIRECTORY = ".keras/java";
    private static final String SYSTEM_KERAS_HOME_VAR = "KERAS_HOME";
    private static final String DEFAULT_KERAS_HOME 
        = Paths.get(System.getProperty("user.home"), KERAS_DIRECTORY).toString();


    public static String home() {
        String systemHome = System.getenv(SYSTEM_KERAS_HOME_VAR);
        return systemHome != null ? systemHome : DEFAULT_KERAS_HOME;
    }

    public static String datasetsDirectory() {
        return Paths.get(home(), "datasets").toString();
    }

    public static Path path(String... path) {
        return Paths.get(home(), path);
    }

    //
    // Keras backend utilties
    //

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

}
