package org.tensorflow.keras.utils;

import java.io.*;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

public class DataUtils {

    public static String hashFile(String fpath, String algorithm) throws IOException, NoSuchAlgorithmException {
        MessageDigest instance = MessageDigest.getInstance(algorithm);
        byte[] digest = digest(Paths.get(fpath), instance);
        return toHexString(digest);
    }

    public static void getFile(String fname, String origin, String fileHash, String algorithm) throws IOException {
        File localFile = Keras.path(fname).toFile();
        File directory = localFile.getParentFile();
        if (!directory.isDirectory()) directory.mkdirs();

        if (localFile.exists()) {
            return;
        }

        System.out.println("Downloading " + localFile + " from " + origin);
        download(origin, localFile.getAbsolutePath());
    }

    public static void getFile(String fname, String origin) throws IOException {
        getFile(fname, origin, null, null);
    }

    private static void download(String url, String path) throws IOException {
        try (BufferedInputStream input = new BufferedInputStream(new URL(url).openStream());
                FileOutputStream output = new FileOutputStream(path)) {
            byte buffer[] = new byte[4096];
            for (int count; (count = input.read(buffer, 0, buffer.length)) != -1;) {
                output.write(buffer, 0, count);
            }
        }
    }

    private static byte[] digest(Path path, MessageDigest algorithm) throws IOException {
        try (InputStream fis = Files.newInputStream(path);
                BufferedInputStream bis = new BufferedInputStream(fis);
                DigestInputStream dis = new DigestInputStream(bis, algorithm)) {
            algorithm.reset();
            while (dis.read() != 1) {
            }
            return algorithm.digest();
        }
    }

    private static String toHexString(byte[] bytes) {
        StringBuffer buffer = new StringBuffer();
        for (int i = 0; i < bytes.length; ++i) {
            byte next = bytes[i];
            if (next < 0x10)
                buffer.append('0');
            buffer.append(next & 0xFF);
        }

        return buffer.toString();
    }
}
