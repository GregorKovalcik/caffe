using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Threading;
using CaffeFeatureExtractorCLR;
using System.Runtime.CompilerServices;

namespace cli_test
{
    class Program
    {
        const int threadCount = 4;
        const int runCount = 10000;
        static Thread[] threads = new Thread[threadCount];
        static Bitmap[] bitmaps = new Bitmap[threadCount];
        static CaffeFeatureExtractor featureExtractor = new CaffeFeatureExtractor("vgg.prototxt", "vgg.caffemodel", "imagenet_mean.binaryproto");
        
        [MethodImpl(MethodImplOptions.Synchronized)]
        static float[] ParseFeature(Bitmap bitmap)
        {
            featureExtractor.SetModeGPU();
            return featureExtractor.ExtractFromImage(bitmap, "fc7");
        }

        static void Main(string[] args)
        {
            featureExtractor.SetModeGPU();
            Bitmap source = (Bitmap)Image.FromFile(0 + ".bmp");
            Bitmap thereAndBackAgain = CaffeFeatureExtractor.TestBitmapConversion(source);
            float[] sourceFeature = ParseFeature(source);

            for (int i = 0; i < threadCount; i++)
            {
                bitmaps[i] = (Bitmap)Image.FromFile(i % 10 + ".bmp");
            }
            for (int i = 0; i < threadCount/2; i++)
            {
                bitmaps[i] = bitmaps[i*2];
            }

            for (int i = 0; i < threadCount; i++)
            {
                int iStable = i;
                threads[iStable] = new Thread(() =>
                {

                    for (int j = 0; j < runCount; j++)
                    {
                        //Bitmap bitmap = (Bitmap)Image.FromFile(iStable + ".bmp");
                        float[] feature = ParseFeature(bitmaps[iStable]);
                        //Console.WriteLine("Thread " + iStable + ": "
                        //    + feature[0] + ", "
                        //    + feature[1] + ", "
                        //    + feature[2] + ", "
                        //    + feature[3] + ", "
                        //    + feature[4] + ", "
                        //    + feature[5] + ", "
                        //    + feature[6] + ", "
                        //    + feature[7] + ", "
                        //    + feature[8]);
                    }
                });
                threads[iStable].Start();

            }

            for (int i = 0; i < threadCount; i++)
            {
                threads[i].Join();
            }
        }
    }
}
