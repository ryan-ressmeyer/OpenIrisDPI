
namespace OpenIris
{
    using Emgu.CV;
    using Emgu.CV.Cvb;
    using Emgu.CV.CvEnum;
    using Emgu.CV.Structure;
    using Emgu.CV.Features2D;
    using System;
    using System.ComponentModel;
    using System.ComponentModel.Composition;
    using System.Drawing;
    using System.Linq;
    using System.Collections.Generic;
    using OpenIris.UI;
    using System.Text;
    using Emgu.CV.Ocl;
    using Emgu.CV.Util;
    using System.Diagnostics;
    using Emgu.CV.Stitching;
    using OpenIris.ImageProcessing;
    using static System.Net.Mime.MediaTypeNames;
    using System.Security.Policy;

    public class Spot
    {
        public RotatedRect rect { get; set; }

        public double mass { get; set; }

        public Spot()
        {
            rect = new RotatedRect();
            mass = 0;
        }
        public Spot(RotatedRect rect, double mass)
        {
            this.rect = rect;
            this.mass = mass;
        }
    }
    public class OpenIrisDPIOutput
    {
        public PointF PupilEst { get; set; }        // Center of mass of thresholded image
        public VectorOfPoint PupilPoints { get; set; }     // Points used to estimate pupil ellipse
        public RotatedRect Pupil { get; set; }        // Estimated Pupil Ellipse
        public Point P1Est { get; set; }       // Center of mass of thresholded pupil
        public RotatedRect P1 { get; set; }          // Refined estimate of P1 CoM
        public Point P4Est { get; set; }       // Maximum pixel after P1 blocked out
        public RotatedRect P4 { get; set; }          // Refined estimate of P4 CoM

        public OpenIrisDPIOutput()
        {
            PupilEst = new PointF(0, 0);
            PupilPoints = new VectorOfPoint();
            Pupil = new RotatedRect();
            P1Est = new Point(0, 0);
            P1 = new RotatedRect();   
            P4Est = new Point(0, 0);
            P4 = new RotatedRect();
        }
    }

    public enum PupilAlgorithm
    {
        Moments,
        EllipseFit,
    }

    public class OpenIrisDPIConfig
    {
        public Rectangle Crop { get; set; }
        public int BlurRadius { get; set; }
        public int PupilThreshold { get; set; }
        public int PupilDsFactor { get; set; }
        public int PupilSearchRadius { get; set; }

        public int PupilFitDsFactor { get; set; }

        public int P1DsFactor { get; set; }
        public int P1Threshold { get; set; }
        public int P1RoiRadius { get; set; }
        public float P1MinDiameter { get; set; }


        public int PupilMaskErodeRadius { get; set; }

        public int P4RoiRadius { get; set; }
        public float P4MinDiameter { get; set; }

        public PupilAlgorithm PupilAlgorithm { get; set; }
    }

    public sealed class OpenIrisDPI : IDisposable
    {
        // Preallocated working iamges
        // (speeds up algorithm assuming parameters are consistent each frame)
        private Mat ImgCrop = new();
        private Mat ImgBlur = new();
        private Mat ImgThresh = new();
        private Mat ImgPupilMasked = new();
        private Mat ImgPupilLap = new();
        private Mat ImgPupilEdge = new();
        private Mat ImgP1 = new();
        private Mat ImgP1Debug = new();
        private Mat ImgP1Ds = new();
        private Mat ImgP1Thresh = new();
        private Mat ImgP4 = new();
        public OpenIrisDPI()
        {

        }
        
        public void Dispose()
        {
            
        }


        public PointF GetCenterOfMass(Mat img, int ds = 1, bool binary = false)
        {
            if (ds < 1) ds = 1;

            Mat imgFit;
            if (ds > 1)
            {
                imgFit = new();
                double f = 1.0 / (double)ds;
                CvInvoke.Resize(img, imgFit, new Size(), f, f, Inter.Nearest);
            }
            else
            {
                imgFit = img;
            }

            Moments moments = CvInvoke.Moments(imgFit, binary);

            double cx = moments.M10 / (moments.M00 + 1e-10) * ds; // x-coordinate of centroid
            double cy = moments.M01 / (moments.M00 + 1e-10) * ds; // y-coordinate of centroid


            return new PointF((float) cx, (float)cy);
        }


        // A function for fitting the second moments of an input image. 
        // If binary is false, is equivalent to finding the best fitting 2d gaussian.
        // If binary is true, is equivalent to finding the best fitting ellipse.
        public Spot Fit2ndMoments(Mat img, int ds = 1, bool binary=false)
        {

            if (ds < 1) ds = 1;
            Mat imgFit;
            if (ds > 1)
            {
                imgFit = new();
                double f = 1.0 / (double)ds;
                CvInvoke.Resize(img, imgFit, new Size(), f, f, Inter.Nearest);
            }
            else
            {
                imgFit = img;
            }

            Moments moments = CvInvoke.Moments(imgFit, binary);

            double M00 = moments.M00 + 1e-10; // added small value to avoid divide by zero error

            double cx = moments.M10 / M00 * ds; // x-coordinate of centroid
            double cy = moments.M01 / M00 * ds; // y-coordinate of centroid

            double mu11 = moments.Mu11 / M00;
            double mu20 = moments.Mu20 / M00;
            double mu02 = moments.Mu02 / M00;

            double theta = 0.5 * Math.Atan2(2 * mu11, (mu20 - mu02)) * 180 / Math.PI + 90; // Rotation angle

            double sqrtTerm = Math.Sqrt(mu11 * mu11 + (mu20 - mu02) * (mu20 - mu02));
            double a = Math.Sqrt(2 * (mu20 + mu02 + sqrtTerm)) * ds; // Semi-major axis
            double b = Math.Sqrt(2 * (mu20 + mu02 - sqrtTerm)) * ds; // Semi-minor axis

            Spot spot = new Spot(
                    new RotatedRect(new PointF((float)cx, (float)cy), new SizeF(2 * (float)a, 2 * (float)b), (float)theta),
                    moments.M00
                );

            return spot;
        }


        public Spot LocalizeSpot(
            Mat img,
            Point roi_center,
            int roi_radius,
            int background_threshold,
            bool block_out)
        {
            Size img_size = img.Size;
            int roi_x = ClipToRange(roi_center.X - roi_radius, 0, img_size.Width);
            int roi_y = ClipToRange(roi_center.Y - roi_radius, 0, img_size.Height);
            int roi_width = ClipToRange(roi_center.X + roi_radius, 0, img_size.Width) - roi_x;
            int roi_height = ClipToRange(roi_center.Y + roi_radius, 0, img_size.Height) - roi_y;

            Rectangle roi = new Rectangle(roi_x, roi_y, roi_width, roi_height);
            var im_roi = new Mat(img, roi);
            var im_roi_float = new Mat();
            im_roi.ConvertTo(im_roi_float, DepthType.Cv32F, 1.0, 0.0);

            var im_roi_sub = im_roi_float - Convert.ToDouble(background_threshold);

            var im_roi_thresh = new Mat();
            CvInvoke.Threshold(im_roi_sub, im_roi_thresh, 0.0, 255.0, ThresholdType.ToZero);

            if (block_out)
            {
                CvInvoke.Rectangle(img, roi, new MCvScalar(0.0), -1, LineType.EightConnected, 0);
            }

            Spot spot = Fit2ndMoments(im_roi_thresh);
            var rect = spot.rect;
            rect.Center.X += roi.X;
            rect.Center.Y += roi.Y;
            spot.rect = rect;
            return spot;
        }

        public int ClipToRange(int value, int minValue, int maxValue)
        {
            return Math.Min(maxValue, Math.Max(value, minValue));
        }

        public Rectangle ClipRect(Rectangle rect, Rectangle boundary)
        {
            int x = Math.Max(rect.X, boundary.X);
            int y = Math.Max(rect.Y, boundary.Y);
            int right = Math.Min(rect.Right, boundary.Right);
            int bottom = Math.Min(rect.Bottom, boundary.Bottom);

            return Rectangle.FromLTRB(x, y, right, bottom); ;
        }
        public Mat DrawFullDebug(OpenIrisDPIOutput output, OpenIrisDPIConfig config)
        {
            Mat colorImg = new Mat();
            CvInvoke.CvtColor(ImgBlur, colorImg, ColorConversion.Gray2Bgr);

            Mat thresholdedColor = new Mat();
            CvInvoke.CvtColor(ImgThresh, thresholdedColor, ColorConversion.Gray2Bgr);

            // Create a blue color Mat with the same size as the original image
            Mat blue = new Mat(ImgBlur.Rows, ImgBlur.Cols, DepthType.Cv8U, 3);
            blue.SetTo(new MCvScalar(70.0, 0.0, 0.0));

            // Perform bitwise AND operation to highlight the thresholded region in blue
            Mat result = new Mat();
            CvInvoke.BitwiseAnd(blue, thresholdedColor, result);

            Mat debug = new Mat();
            CvInvoke.Add(colorImg, result, debug);

            // Draw Search Radius
            CvInvoke.Circle(
                debug,
                new Point((int) output.PupilEst.X, (int) output.PupilEst.Y),
                config.PupilSearchRadius,
                new MCvScalar(255, 255, 255),
                1,
                LineType.EightConnected,
                0
            );

            bool p1_valid = output.P1.Center.X >= 0;
            if ( p1_valid )
            {
                // Draw P1 
                CvInvoke.Circle(
                    debug,
                    new Point((int)output.P1.Center.X, (int)output.P1.Center.Y),
                    config.P1RoiRadius,
                    new MCvScalar(0, 255, 255),
                    1,
                    LineType.AntiAlias,
                    0
                );

                // Draw P1 Roi
                var p1Loc = output.P1Est;
                p1Loc.X -= config.P1RoiRadius;
                p1Loc.Y -= config.P1RoiRadius;
                var p1Size = new Size(config.P1RoiRadius * 2, config.P1RoiRadius * 2);
                CvInvoke.Rectangle(
                    debug,
                    new Rectangle(p1Loc, p1Size),
                    new MCvScalar(0, 255, 255),
                    1,
                    LineType.AntiAlias,
                    0
                );
            }
            

            // Draw p4
            bool p4_valid = output.P4.Center.Y >= 0;
            if ( p4_valid ) 
            {
            CvInvoke.Circle(
                    debug,
                    new Point((int)output.P4.Center.X, (int)output.P4.Center.Y),
                    config.P4RoiRadius,
                    new MCvScalar(0, 0, 255),
                    1,
                    LineType.AntiAlias,
                    0
                );

                var p4Loc = output.P4Est;
                p4Loc.X -= config.P4RoiRadius;
                p4Loc.Y -= config.P4RoiRadius;
                var p4Size = new Size(config.P4RoiRadius * 2, config.P4RoiRadius * 2);
                // Draw P4 ROI and P4
                CvInvoke.Rectangle(
                    debug,
                    new Rectangle(p4Loc, p4Size),
                    new MCvScalar(0, 0, 255),
                    1,
                    LineType.AntiAlias,
                    0
                ); 
            }

            // Draw pupil outline
            CvInvoke.Circle(debug,
                new Point((int)output.Pupil.Center.X, (int)output.Pupil.Center.Y), 2, new MCvScalar(0, 255, 0), -1);
            if (config.PupilAlgorithm == PupilAlgorithm.Moments)
            {
                if (output.PupilPoints.Size > 0)
                {
                    CvInvoke.DrawContours(debug, new VectorOfVectorOfPoint(output.PupilPoints), 0, new MCvScalar(0, 255, 0));
                }
            } else if (config.PupilAlgorithm == PupilAlgorithm.EllipseFit)
            {
                for (int i = 0; i < output.PupilPoints.Size; i++)
                {
                    CvInvoke.Circle(debug, output.PupilPoints[i], 1, new MCvScalar(0, 255, 0), -1);
                }
            }

            return debug;
        }

        public Mat DrawPupilDebug(OpenIrisDPIOutput output, OpenIrisDPIConfig config)
        {
            Mat pupilEdge = new();
            CvInvoke.CvtColor(ImgPupilEdge, pupilEdge, ColorConversion.Gray2Bgr);

            var pupilSearchOffset = new Point((int)output.PupilEst.X - config.PupilSearchRadius, (int)output.PupilEst.Y - config.PupilSearchRadius);
            var pupilSearchRect = new Rectangle(pupilSearchOffset, new Size(config.PupilSearchRadius * 2, config.PupilSearchRadius * 2));
            pupilSearchRect = ClipRect(pupilSearchRect, new Rectangle(new Point(0, 0), ImgThresh.Size));
            pupilSearchOffset = pupilSearchRect.Location;

            var debug = pupilEdge.ToImage<Bgr, byte>();

            if (config.PupilAlgorithm == PupilAlgorithm.Moments)
            {
                if (output.PupilPoints.Size > 0)
                {
                    var pupilContour = new List<Point>(output.PupilPoints.Size);
                    for (int i = 0; i < output.PupilPoints.Size; i++)
                    {
                        var point = output.PupilPoints[i];
                        pupilContour.Add(new Point(point.X - pupilSearchOffset.X, point.Y - pupilSearchOffset.Y));
                    }
                    var pupilContourVec = new VectorOfPoint(pupilContour.ToArray());
                    CvInvoke.DrawContours(debug, new VectorOfVectorOfPoint(pupilContourVec), 0, new MCvScalar(0, 255, 255));
                }
            } else if (config.PupilAlgorithm == PupilAlgorithm.EllipseFit)
            {
                for (int i = 0; i < output.PupilPoints.Size; i++)
                {
                    var point = output.PupilPoints[i];
                    CvInvoke.Circle(debug, new Point(point.X - pupilSearchOffset.X, point.Y - pupilSearchOffset.Y), 1, new MCvScalar(0, 255, 255), -1);
                    
                }
            }
            
            var ell = output.Pupil;
            ell.Center -= new Size(pupilSearchOffset);
            CvInvoke.Circle(debug, new Point((int) ell.Center.X, (int) ell.Center.Y), 2, new MCvScalar(0, 255, 0), -1);
            CvInvoke.Ellipse(debug, ell, new MCvScalar(0, 255, 0), 1);
 
            return debug.Mat;
        }

        public Mat DrawP1Debug(OpenIrisDPIOutput output, OpenIrisDPIConfig config)
        {
            Mat colorImg = new();
            CvInvoke.CvtColor(ImgP1Debug, colorImg, ColorConversion.Gray2Bgr);

            Mat p1Thresh = new();
            CvInvoke.Threshold(ImgP1Debug, p1Thresh, config.P1Threshold, 255, ThresholdType.Binary);
            CvInvoke.CvtColor(p1Thresh, p1Thresh, ColorConversion.Gray2Bgr);

            // Create a blue color Mat with the same size as the original image
            Mat color = new Mat(p1Thresh.Rows, p1Thresh.Cols, DepthType.Cv8U, 3);
            color.SetTo(new MCvScalar(0.0, 120.0, 120.0));

            // Perform bitwise AND operation to highlight the thresholded region in blue
            CvInvoke.BitwiseAnd(color, p1Thresh, p1Thresh);

            // Add the mats together to get a debug image
            Mat debug = colorImg / 2 + p1Thresh;

            var pupilSearchOffset = new Point((int)output.PupilEst.X - config.PupilSearchRadius, (int)output.PupilEst.Y - config.PupilSearchRadius);
            var pupilSearchRect = new Rectangle(pupilSearchOffset, new Size(config.PupilSearchRadius * 2, config.PupilSearchRadius * 2));
            pupilSearchRect = ClipRect(pupilSearchRect, new Rectangle(new Point(0, 0), ImgThresh.Size));
            pupilSearchOffset = pupilSearchRect.Location;

            // Draw P1 ROI and Point
            var p1Loc = output.P1Est - ((Size)pupilSearchOffset);
            p1Loc.X -= config.P1RoiRadius;
            p1Loc.Y -= config.P1RoiRadius;
            var p1Size = new Size(config.P1RoiRadius * 2, config.P1RoiRadius * 2);
            CvInvoke.Rectangle(
                debug,
                new Rectangle(p1Loc, p1Size),
                new MCvScalar(0, 255, 255),
                1,
                LineType.AntiAlias,
                0
            );

            CvInvoke.Circle(
                debug,
                new Point((int)output.P1.Center.X, (int)output.P1.Center.Y) - ((Size)pupilSearchOffset),
                config.P1RoiRadius,
                new MCvScalar(0, 255, 255),
                1,
                LineType.AntiAlias,
                0
            );

            return debug;
        }

        public Mat DrawP4Debug(OpenIrisDPIOutput output, OpenIrisDPIConfig config)
        {
            Mat colorImg = new();
            CvInvoke.CvtColor(ImgP4, colorImg, ColorConversion.Gray2Bgr);

            Mat p4Thresh = new();
            CvInvoke.Threshold(ImgP4, p4Thresh, config.PupilThreshold, 255, ThresholdType.Binary);
            CvInvoke.CvtColor(p4Thresh, p4Thresh, ColorConversion.Gray2Bgr);

            // Create a yellow color Mat with the same size as the original image
            Mat color = new Mat(p4Thresh.Rows, p4Thresh.Cols, DepthType.Cv8U, 3);
            color.SetTo(new MCvScalar(120.0, 0.0, 120.0));

            // Perform bitwise AND operation to highlight the thresholded region in blue
            CvInvoke.BitwiseAnd(color, p4Thresh, p4Thresh);

            // Add the mats together to get a debug image
            Mat debug = colorImg / 2 + p4Thresh;

            var pupilSearchOffset = new Point((int)output.PupilEst.X - config.PupilSearchRadius, (int)output.PupilEst.Y - config.PupilSearchRadius);
            var pupilSearchRect = new Rectangle(pupilSearchOffset, new Size(config.PupilSearchRadius * 2, config.PupilSearchRadius * 2));
            pupilSearchRect = ClipRect(pupilSearchRect, new Rectangle(new Point(0, 0), ImgThresh.Size));
            pupilSearchOffset = pupilSearchRect.Location;

            var p4Loc = output.P4Est - ((Size)pupilSearchOffset);
            p4Loc.X -= config.P4RoiRadius;
            p4Loc.Y -= config.P4RoiRadius;
            var p4Size = new Size(config.P4RoiRadius * 2, config.P4RoiRadius * 2);
            // Draw P4 ROI and P4
            CvInvoke.Rectangle(
                debug,
                new Rectangle(p4Loc, p4Size),
                new MCvScalar(0, 0, 255),
                1,
                LineType.AntiAlias,
                0
            );

            CvInvoke.Circle(
                debug,
                new Point((int)output.P4.Center.X, (int)output.P4.Center.Y) - ((Size)pupilSearchOffset),
                config.P4RoiRadius,
                new MCvScalar(0, 0, 255),
                1,
                LineType.AntiAlias,
                0
            );

            return debug;
        }


        public OpenIrisDPIOutput FindDualPurkinje(ImageEye imageEye, OpenIrisDPIConfig config)
        {
            // need to stash DEBUG in case it changes in the middle of a loop
            bool debug = EyeTracker.DEBUG;

            var output = new OpenIrisDPIOutput();

            if (imageEye.Size.IsEmpty)
                return output;

            // ########################################
            // Estimate Pupil Center 
            // ########################################

            EyeTrackerDebug.TrackProcessingTime("Estimate Pupil Center");

            ImgCrop = new Mat(imageEye.Image.Mat, config.Crop);

            // Preprocess image by cropping and bluring
            CvInvoke.GaussianBlur(ImgCrop, ImgBlur, new Size(config.BlurRadius * 2 + 1, config.BlurRadius * 2 + 1), 0, 0, BorderType.Default);

            // Threshold image to find pupil
            CvInvoke.Threshold(ImgBlur, ImgThresh, config.PupilThreshold, 255, ThresholdType.BinaryInv);

            // Downsample for speedup, take center of mass of dark pixels to extimate pupil center
            output.PupilEst = GetCenterOfMass(ImgThresh, config.PupilDsFactor, true);


            // ########################################
            // Fit Pupil 
            // ########################################

            EyeTrackerDebug.TrackProcessingTime("Fitting Pupil");

            // Crop out approximate pupil location
            var pupilSearchOffset = new Point((int)output.PupilEst.X - config.PupilSearchRadius, (int)output.PupilEst.Y - config.PupilSearchRadius);
            var pupilSearchRect = new Rectangle(pupilSearchOffset, new Size(config.PupilSearchRadius * 2, config.PupilSearchRadius * 2));
            pupilSearchRect = ClipRect(pupilSearchRect, new Rectangle(new Point(0, 0), ImgThresh.Size));
            pupilSearchOffset = pupilSearchRect.Location;
            var pupilSearchROI = new Mat(ImgThresh, pupilSearchRect);

            // Mask in a circle of the search radius
            var pupilSearchMask = new Mat(pupilSearchROI.Size, DepthType.Cv8U, 1);
            pupilSearchMask.SetTo(new MCvScalar(0));
            var pupilSearchCenter = new Point((int)output.PupilEst.X - pupilSearchOffset.X, (int)output.PupilEst.Y - pupilSearchOffset.Y);
            CvInvoke.Circle(pupilSearchMask, pupilSearchCenter, config.PupilSearchRadius, new MCvScalar(255), -1, LineType.EightConnected, 0);
            CvInvoke.BitwiseAnd(pupilSearchROI, pupilSearchMask, ImgPupilMasked);

            // Find edges of pupil by computing the laplacian filter of the thresholded image
            CvInvoke.Laplacian(ImgPupilMasked, ImgPupilLap, DepthType.Cv16S, 1, 1, 0, BorderType.Constant);
            CvInvoke.ConvertScaleAbs(ImgPupilLap, ImgPupilEdge, 1.0, 0.0);
            ImgPupilEdge.ConvertTo(ImgPupilEdge, DepthType.Cv8U);

            var edgePts = new VectorOfPoint();
            CvInvoke.FindNonZero(ImgPupilEdge, edgePts);

            // Pupil Fitting
            RotatedRect pupil = new RotatedRect();
            var pupilPtsRaw = new VectorOfPoint();
            if (config.PupilAlgorithm == PupilAlgorithm.Moments)
            {
                var pupilRaw = new RotatedRect();
                var pupilBitmask = Mat.Zeros(ImgPupilEdge.Rows, ImgPupilEdge.Cols, DepthType.Cv8U, 1);
                if (edgePts.Size != 0)
                {
                    CvInvoke.ConvexHull(edgePts, pupilPtsRaw, false);
                    CvInvoke.DrawContours(pupilBitmask, new VectorOfVectorOfPoint(pupilPtsRaw), 0, new MCvScalar(255), -1);

                    var pupil_spot = Fit2ndMoments(pupilBitmask, config.PupilFitDsFactor, true);
                    pupilRaw = pupil_spot.rect;
                }

                pupil = new RotatedRect(
                    new PointF(pupilRaw.Center.X + pupilSearchOffset.X, pupilRaw.Center.Y + pupilSearchOffset.Y),
                    pupilRaw.Size, pupilRaw.Angle);

                output.Pupil = pupil;

                var pupilPtsArr = pupilPtsRaw.ToArray();
                for (int i = 0; i < pupilPtsRaw.Size; i++)
                {
                    Point point = pupilPtsArr[i];
                    point.X += pupilSearchOffset.X;
                    point.Y += pupilSearchOffset.Y;
                    pupilPtsArr[i] = point;
                }

                var pupilPts = new VectorOfPoint(pupilPtsArr);
                output.PupilPoints = pupilPts;
            } else if (config.PupilAlgorithm == PupilAlgorithm.EllipseFit)
            {
                if (edgePts.Size != 0)
                {
                    CvInvoke.ConvexHull(edgePts, pupilPtsRaw, false);
                    var pupilContour = Mat.Zeros(ImgPupilEdge.Rows, ImgPupilEdge.Cols, DepthType.Cv8U, 1);
                    CvInvoke.DrawContours(pupilContour, new VectorOfVectorOfPoint(pupilPtsRaw), 0, new MCvScalar(255));
                    // Mask out edges
                    CvInvoke.Rectangle(pupilContour, new Rectangle(0, 0, pupilContour.Width - 1, pupilContour.Height - 1), new MCvScalar(0));
                    CvInvoke.FindNonZero(pupilContour, pupilPtsRaw);
                }

                var pupilPtsArr = pupilPtsRaw.ToArray();
                for (int i = 0; i < pupilPtsRaw.Size; i++)
                {
                    Point point = pupilPtsArr[i];
                    point.X += pupilSearchOffset.X;
                    point.Y += pupilSearchOffset.Y;
                    pupilPtsArr[i] = point;
                }

                var pupilPts = new VectorOfPoint(pupilPtsArr);

                output.PupilPoints = pupilPts;

                if (pupilPts.Size >= 5)
                {
                    // NOTE: pupil rotation not being assigned correctly in this version of OpenCV. see https://github.com/opencv/opencv/issues/11088
                    pupil = CvInvoke.FitEllipse(pupilPts);
                }
                else
                {
                    // TODO retrun error code here
                    pupil = new RotatedRect(new PointF(0, 0), new SizeF(0, 0), 0);
                }
                output.Pupil = pupil;
            }
            

            // ########################################
            // Find P1 (potentially outside of pupil)
            // ########################################

            EyeTrackerDebug.TrackProcessingTime("Finding P1");

            // Mask out everything outside of the search area
            var p1MaskCenter = pupil.Center;
            p1MaskCenter.X -= pupilSearchOffset.X;
            p1MaskCenter.Y -= pupilSearchOffset.Y;

            var p1MaskSize = new SizeF { Height = config.PupilSearchRadius*2, Width = config.PupilSearchRadius*2};

            var p1MaskEllipse = new RotatedRect(p1MaskCenter, p1MaskSize, 0);

            var p1Mask = new Mat(pupilSearchRect.Size, DepthType.Cv8U, 1);
            p1Mask.SetTo(new MCvScalar(0));
            CvInvoke.Ellipse(p1Mask, p1MaskEllipse, new MCvScalar(255), -1, LineType.EightConnected);

            var p1ROI = new Mat(ImgBlur, pupilSearchRect);

            ImgP1.SetTo(new MCvScalar(0));
            CvInvoke.BitwiseAnd(p1ROI, p1ROI, ImgP1, p1Mask);

            if (debug)
            {
                ImgP1.CopyTo(ImgP1Debug);
            }

            var imP1DsSize = new Size(pupilSearchRect.Width / config.P1DsFactor, pupilSearchRect.Height / config.P1DsFactor);
            CvInvoke.Resize(ImgP1, ImgP1Ds, imP1DsSize, 0, 0, Inter.Linear);
            CvInvoke.Threshold(ImgP1Ds, ImgP1Thresh, config.P1Threshold, 255, ThresholdType.Binary);

            var p1COM = GetCenterOfMass(ImgP1Thresh);
            p1COM.X *= config.P1DsFactor;
            p1COM.Y *= config.P1DsFactor;

            output.P1Est = new Point((int)p1COM.X + pupilSearchOffset.X, (int)p1COM.Y + pupilSearchOffset.Y);

            var p1_spot = LocalizeSpot(ImgP1, new Point((int)p1COM.X, (int)p1COM.Y), config.P1RoiRadius, config.P1Threshold, true);
            var p1 = p1_spot.rect;
            p1.Center.X += pupilSearchOffset.X;
            p1.Center.Y += pupilSearchOffset.Y;

            // If semi-major axis of p1 is too small (i.e. P1 not detected) set to invalid location
            bool p1_valid = p1.Size.Height >= config.P1MinDiameter;
            if (!p1_valid) p1.Center = new PointF(-100, -100);

            output.P1 = p1;


            // ########################################
            // Find P4 (always within pupil)
            // ########################################

            EyeTrackerDebug.TrackProcessingTime("Finding P4");

            // If the pupil isn't found, assume the search radius is too small and just look within the P1 ROI.
            if (pupilPtsRaw.Size >= 3)
            {
                //var p4Mask = pupilBitmask;

                
                var p4Mask = Mat.Zeros(pupilSearchRect.Size.Height, pupilSearchRect.Size.Width, DepthType.Cv8U, 1);

                if (pupilPtsRaw.Size > 0)
                {
                    CvInvoke.DrawContours(p4Mask, new VectorOfVectorOfPoint(pupilPtsRaw), 0, new MCvScalar(255), -1);
                }

                if (config.PupilMaskErodeRadius > 0)
                {
                    var kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(config.PupilMaskErodeRadius * 2 + 1, config.PupilMaskErodeRadius * 2 + 1), new Point(-1, -1));
                    CvInvoke.Erode(p4Mask, p4Mask, kernel, new Point(-1, -1), 1, BorderType.Constant, new MCvScalar(1));
                }

                ImgP4.SetTo(new MCvScalar(0));
                CvInvoke.BitwiseAnd(ImgP1, ImgP1, ImgP4, p4Mask);
            } else
            {
                ImgP1.CopyTo(ImgP4);
            }

            // Find P4
            Point maxLoc = new(), minLoc = new();
            double minVal = 0.0, maxVal = 0.0;
            CvInvoke.MinMaxLoc(ImgP4, ref maxVal, ref minVal, ref minLoc, ref maxLoc);
            output.P4Est = new Point(maxLoc.X + pupilSearchOffset.X, maxLoc.Y + pupilSearchOffset.Y);

            var p4_spot = LocalizeSpot(ImgP4, maxLoc, config.P4RoiRadius, config.PupilThreshold, false);
            var p4 = p4_spot.rect;
            p4.Center.X += pupilSearchOffset.X;
            p4.Center.Y += pupilSearchOffset.Y;

            // If p4 not large enough (i.e. not found) set to invalid position
            bool p4_valid = p4.Size.Height >= config.P4MinDiameter;
            if (!p4_valid || !p1_valid) p4.Center = new PointF(config.Crop.Width + 100, -100);

            output.P4 = p4;


            // ########################################
            // Debugging
            // ########################################

            EyeTrackerDebug.TrackProcessingTime("Debugging");

            if (debug)
            {
                var debugFulMat = DrawFullDebug(output, config);
                EyeTrackerDebug.AddImage("DPI-Overview", imageEye.WhichEye, debugFulMat.ToImage<Bgr, byte>());

                var debugPupil = DrawPupilDebug(output, config);
                EyeTrackerDebug.AddImage("DPI-Pupil", imageEye.WhichEye, debugPupil.ToImage<Bgr, byte>());

                var debugP1Mat = DrawP1Debug(output, config);
                EyeTrackerDebug.AddImage("DPI-P1", imageEye.WhichEye, debugP1Mat.ToImage<Bgr, byte>());

                var debugP4Mat = DrawP4Debug(output, config);
                EyeTrackerDebug.AddImage("DPI-P4", imageEye.WhichEye, debugP4Mat.ToImage<Bgr, byte>());

            }

            return output;
        }

    }
    
}
