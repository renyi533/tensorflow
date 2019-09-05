package graph_data_parser;

import java.nio.ByteBuffer;

class Bytes {
      public static byte[] changeBytes(byte[] a) {
        byte[] b = new byte[a.length];
        for (int i = 0; i < b.length; i++) {
          b[i] = a[b.length - i - 1];
        }
        return b;
      }

      public static int bytesToInt(byte[] bytes) {
        return ByteBuffer.wrap(bytes).getInt();
      }

      public static byte[] intToBytes(int value) {
        return ByteBuffer.allocate(4).putInt(value).array();
      }

      public static byte[] floatToBytes(float value) {
        return ByteBuffer.allocate(4).putFloat(value).array();
      }

      public static float bytesToFloat(byte[] bytes) {
        return ByteBuffer.wrap(bytes).getFloat();
      }

      public static byte[] longToBytes(long value) {
        return ByteBuffer.allocate(8).putLong(value).array();
      }

      public static long bytesToLong(byte[] bytes) {
        return ByteBuffer.wrap(bytes).getLong();
      }
}

