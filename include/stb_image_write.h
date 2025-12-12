#ifndef STB_IMAGE_WRITE_H
#define STB_IMAGE_WRITE_H

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static unsigned char *stbiw__zlib_compress(unsigned char *data, int data_len,
                                           int *out_len, int quality);

static void stbiw__putc(FILE *f, int c) { fputc(c, f); }

static void stbiw__write3(FILE *f, unsigned char a, unsigned char b,
                          unsigned char c) {
  fputc(a, f);
  fputc(b, f);
  fputc(c, f);
}

int stbi_write_ppm(char const *filename, int w, int h, int comp,
                   const void *data) {
  FILE *f = fopen(filename, "wb");
  if (!f)
    return 0;
  fprintf(f, "P6\n%d %d\n255\n", w, h);
  const unsigned char *d = (const unsigned char *)data;
  for (int j = 0; j < h; ++j) {
    for (int i = 0; i < w; ++i) {
      int idx = (j * w + i) * comp;
      if (comp >= 3) {
        fputc(d[idx + 0], f);
        fputc(d[idx + 1], f);
        fputc(d[idx + 2], f);
      } else {
        fputc(d[idx], f);
        fputc(d[idx], f);
        fputc(d[idx], f);
      }
    }
  }
  fclose(f);
  return 1;
}

int stbi_write_bmp(char const *filename, int w, int h, int comp,
                   const void *data) {
  FILE *f = fopen(filename, "wb");
  if (!f)
    return 0;

  int pad = (-w * 3) & 3;
  int filesize = 54 + (w * 3 + pad) * h;

  unsigned char header[54] = {0};
  header[0] = 'B';
  header[1] = 'M';
  header[2] = filesize;
  header[3] = filesize >> 8;
  header[4] = filesize >> 16;
  header[5] = filesize >> 24;
  header[10] = 54;
  header[14] = 40;
  header[18] = w;
  header[19] = w >> 8;
  header[20] = w >> 16;
  header[21] = w >> 24;
  header[22] = h;
  header[23] = h >> 8;
  header[24] = h >> 16;
  header[25] = h >> 24;
  header[26] = 1;
  header[28] = 24;

  fwrite(header, 1, 54, f);

  const unsigned char *d = (const unsigned char *)data;
  unsigned char padBytes[3] = {0, 0, 0};

  for (int j = h - 1; j >= 0; --j) {
    for (int i = 0; i < w; ++i) {
      int idx = (j * w + i) * comp;
      unsigned char b = comp >= 3 ? d[idx + 2] : d[idx];
      unsigned char g = comp >= 3 ? d[idx + 1] : d[idx];
      unsigned char r = comp >= 3 ? d[idx + 0] : d[idx];
      fputc(b, f);
      fputc(g, f);
      fputc(r, f);
    }
    if (pad)
      fwrite(padBytes, 1, pad, f);
  }

  fclose(f);
  return 1;
}

#endif // STB_IMAGE_WRITE_H
