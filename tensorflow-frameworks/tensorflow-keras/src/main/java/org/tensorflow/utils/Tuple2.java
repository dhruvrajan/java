package org.tensorflow.utils;

import org.tensorflow.data.Pair;

public class Tuple2<T> extends Pair<T, T> {
    public Tuple2(T first, T second){
        super(first, second);
    }
}
