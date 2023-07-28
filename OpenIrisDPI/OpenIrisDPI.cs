
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

    public class OpenIrisDPIOutput
    {
        public PointF PupilEst { get; set; }        // Center of mass of thresholded image
        public VectorOfPoint PupilPoints { get; set; }     // Points used to estimate pupil ellipse
        public RotatedRect Pupil { get; set; }        // Estimated Pupil Ellipse
        public Point P1Est { get; set; }       // Center of mass of thresholded pupil
        public PointF P1 { get; set; }          // Refined estimate of P1 CoM
        public Point P4Est { get; set; }       // Maximum pixel after P1 blocked out
        public PointF P4 { get; set; }          // Refined estimate of P4 CoM

        public OpenIrisDPIOutput()
        {
            PupilPoints = new VectorOfPoint();
        }
    }

    public class OpenIrisDPIConfig
    {
        public Rectangle Crop { get; set; }
        public int BlurSize { get; set; }
        public int PupilThreshold { get; set; }
        public int PupilDsFactor { get; set; }
        public int PupilSearchRadius { get; set; }

        public float PupilMaskPercent { get; set; }
        public int P1DsFactor { get; set; }
        public int P1Threshold { get; set; }
        public int P1RoiRadius { get; set; }

        public int P4Threshold { get; set; }

        public int P4RoiRadius { get; set; }
    }

    public sealed class OpenIrisDPI : IDisposable
    {
        // Preallocated working iamges
        // (speeds up algorithm assuming parameters are consistent each frame)
        private Mat ImgCrop = new();
        private Mat ImgBlur = new();
        private Mat ImgThresh = new();
        private Mat ImgThreshDs = new();
        private Mat ImgPupilEdge = new();
        private Mat ImgPupilEdgeMasked = new();
        private Mat ImgPupil = new();
        private Mat ImgPupilDebug = new();
        private Mat ImgP1Ds = new();
        private Mat ImgP1Thresh = new();

        public PointF GetCenterOfMass(Mat img)
        {

            Moments m = CvInvoke.Moments(img);

            return new PointF(
                Convert.ToSingle(m.M10 / (m.M00 + 1e-8)),
                Convert.ToSingle(m.M01 / (m.M00 + 1e-8))
            );
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

        public PointF LocalizeSpot(
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

            var im_roi_power = new Mat();
            CvInvoke.Pow(im_roi_thresh, 5.0, im_roi_power);

            if (block_out)
            {
                CvInvoke.Rectangle(img, roi, new MCvScalar(0.0), -1, LineType.EightConnected, 0);
            }

            PointF com = GetCenterOfMass(im_roi_power);
            com.X += roi.X;
            com.Y += roi.Y;
            return com;
        }

        public Mat DrawFullDebug(OpenIrisDPIOutput output, OpenIrisDPIConfig config)
        {
            Mat colorImg = new Mat();
            CvInvoke.CvtColor(ImgBlur, colorImg, ColorConversion.Gray2Bgr);

            Mat thresholdedColor = new Mat();
            CvInvoke.CvtColor(ImgThresh, thresholdedColor, ColorConversion.Gray2Bgr);

            // Create a blue color Mat with the same size as the original image
            Mat blue = new Mat(ImgBlur.Rows, ImgBlur.Cols, DepthType.Cv8U, 3);
            blue.SetTo(new MCvScalar(120.0, 0.0, 0.0));

            // Perform bitwise AND operation to highlight the thresholded region in blue
            Mat result = new Mat();
            CvInvoke.BitwiseAnd(blue, thresholdedColor, result);

            Mat debug = new Mat();
            CvInvoke.Add(colorImg, result, debug);

            CvInvoke.Circle(
                debug,
                new Point((int) output.PupilEst.X, (int) output.PupilEst.Y),
                config.PupilSearchRadius,
                new MCvScalar(255, 0, 0),
                2,
                LineType.EightConnected,
                0
            );

            // Draw P1 ROI and P1
            CvInvoke.Circle(
                debug,
                new Point((int) output.P1.X, (int) output.P1.Y),
                3,
                new MCvScalar(0, 0, 255),
                -1,
                LineType.AntiAlias,
                0
            );

            var p1Loc = output.P1Est;
            p1Loc.X -= config.P1RoiRadius;
            p1Loc.Y -= config.P1RoiRadius;
            var p1Size = new Size(config.P1RoiRadius * 2, config.P1RoiRadius * 2);
            CvInvoke.Rectangle(
                debug,
                new Rectangle(p1Loc, p1Size),
                new MCvScalar(255, 255, 255),
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
                new MCvScalar(255, 0, 255),
                1,
                LineType.AntiAlias,
                0
            );

            CvInvoke.Circle(
                debug,
                new Point((int) output.P4.X, (int) output.P4.Y),
                2,
                new MCvScalar(0, 0, 255),
                -1,
                LineType.AntiAlias,
                0
            );

            CvInvoke.Ellipse(
                debug,
                output.Pupil,
                new MCvScalar(0, 255, 0),
                1,
                LineType.AntiAlias
            );

            foreach (Point p in output.PupilPoints.ToArray())
            {
                CvInvoke.Circle(
                    debug,
                    p,
                    2,
                    new MCvScalar(0, 255, 255),
                    -1,
                    LineType.AntiAlias,
                    0
                );
            }

            return debug;
        }


        public Mat DrawPupilDebug(OpenIrisDPIOutput output, OpenIrisDPIConfig config)
        {
            Mat pupilEdge = new();
            CvInvoke.CvtColor(ImgPupilEdgeMasked, pupilEdge, ColorConversion.Gray2Bgr);

            var pupilSearchOffset = new Point((int)output.PupilEst.X - config.PupilSearchRadius, (int)output.PupilEst.Y - config.PupilSearchRadius);
            var pupilSearchRect = new Rectangle(pupilSearchOffset, new Size(config.PupilSearchRadius * 2, config.PupilSearchRadius * 2));
            pupilSearchRect = ClipRect(pupilSearchRect, new Rectangle(new Point(0, 0), ImgThresh.Size));
            pupilSearchOffset = pupilSearchRect.Location;

            var imgDebug = pupilEdge.ToImage<Bgr, byte>();
            for (int i = 0; i < output.PupilPoints.Size; i++)
            {
                var point = output.PupilPoints[i];
                var row = point.Y - pupilSearchOffset.Y;
                var col = point.X - pupilSearchOffset.X;
                imgDebug[row, col] = new Bgr(0, 255, 255);
            }
            var debug = imgDebug;

            var ell = output.Pupil;
            ell.Center -= new Size(pupilSearchOffset);
            CvInvoke.Ellipse(debug, ell, new MCvScalar(0, 255, 0), 1);

            return debug.Mat;
        }

        public Mat DrawP1P4Debug(OpenIrisDPIOutput output, OpenIrisDPIConfig config)
        {
            Mat colorImg = new();
            CvInvoke.CvtColor(ImgPupilDebug, colorImg, ColorConversion.Gray2Bgr);

            Mat p1Thresh = new();
            CvInvoke.Threshold(ImgPupilDebug, p1Thresh, config.P1Threshold, 255, ThresholdType.Binary);
            CvInvoke.CvtColor(p1Thresh, p1Thresh, ColorConversion.Gray2Bgr);

            // Create a yellow color Mat with the same size as the original image
            Mat yellow = new Mat(p1Thresh.Rows, p1Thresh.Cols, DepthType.Cv8U, 3);
            yellow.SetTo(new MCvScalar(0.0, 80.0, 80.0));

            // Perform bitwise AND operation to highlight the thresholded region in blue
            CvInvoke.BitwiseAnd(yellow, p1Thresh, p1Thresh);

            Mat p4Thresh = new();
            CvInvoke.Threshold(ImgPupilDebug, p4Thresh, config.P4Threshold, 255, ThresholdType.Binary);
            CvInvoke.CvtColor(p4Thresh, p4Thresh, ColorConversion.Gray2Bgr);

            // Create a magenta color Mat with the same size as the original image
            Mat magenta = new Mat(p4Thresh.Rows, p4Thresh.Cols, DepthType.Cv8U, 3);
            magenta.SetTo(new MCvScalar(120.0, 0.0, 120.0));

            // Perform bitwise AND operation to highlight the thresholded region in blue
            CvInvoke.BitwiseAnd(magenta, p4Thresh, p4Thresh);

            // Add the mats together to get a debug image
            Mat debug = colorImg / 2 + p1Thresh + p4Thresh;

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
                new MCvScalar(0, 0, 255),
                1,
                LineType.AntiAlias,
                0
            );

            CvInvoke.Circle(
                debug,
                new Point((int)output.P1.X, (int)output.P1.Y) - ((Size)pupilSearchOffset),
                2,
                new MCvScalar(0, 0, 255),
                -1,
                LineType.AntiAlias,
                0
            );

            var p4Loc = output.P4Est - ((Size)pupilSearchOffset);
            p4Loc.X -= config.P4RoiRadius;
            p4Loc.Y -= config.P4RoiRadius;
            var p4Size = new Size(config.P4RoiRadius * 2, config.P4RoiRadius * 2);
            // Draw P4 ROI and P4
            CvInvoke.Rectangle(
                debug,
                new Rectangle(p4Loc, p4Size),
                new MCvScalar(0, 255, 255),
                1,
                LineType.AntiAlias,
                0
            );

            CvInvoke.Circle(
                debug,
                new Point((int)output.P4.X, (int)output.P4.Y) - ((Size)pupilSearchOffset),
                2,
                new MCvScalar(0, 255, 255),
                -1,
                LineType.AntiAlias,
                0
            );

            return debug;
        }

        public OpenIrisDPIOutput FindDualPurkinje(ImageEye imageEye, OpenIrisDPIConfig config)
        {
            var output = new OpenIrisDPIOutput();


            ImgCrop = new Mat(imageEye.Image.Mat, config.Crop);

            CvInvoke.GaussianBlur(ImgCrop, ImgBlur, new Size(config.BlurSize, config.BlurSize), 0, 0, BorderType.Default);
            CvInvoke.Threshold(ImgBlur, ImgThresh, config.PupilThreshold, 255, ThresholdType.BinaryInv);

            var imThreshDsSize = new Size(config.Crop.Width / config.PupilDsFactor, config.Crop.Height / config.PupilDsFactor);
            CvInvoke.Resize(ImgThresh, ImgThreshDs, imThreshDsSize, 0, 0, Inter.Linear);

            var com = GetCenterOfMass(ImgThreshDs);
            com.X *= config.PupilDsFactor;
            com.Y *= config.PupilDsFactor;
            output.PupilEst = com;

            EyeTrackerDebug.TrackProcessingTime("Pupil CoM");
            // Find edges of pupil by computing the sobel filter of the thresholded image

            var pupilSearchOffset = new Point((int) com.X - config.PupilSearchRadius, (int) com.Y - config.PupilSearchRadius);
            var pupilSearchRect = new Rectangle(pupilSearchOffset, new Size(config.PupilSearchRadius * 2, config.PupilSearchRadius * 2));
            pupilSearchRect = ClipRect(pupilSearchRect, new Rectangle(new Point(0, 0), ImgThresh.Size));
            pupilSearchOffset = pupilSearchRect.Location;
            var pupilSearchROI = new Mat(ImgThresh, pupilSearchRect);

            CvInvoke.Sobel(pupilSearchROI, ImgPupilEdge, DepthType.Cv8U, 1, 1);

            // Mask in a circle of the search radius
            var pupilSearchMask = new Mat(pupilSearchROI.Size, DepthType.Cv8U, 1);
            pupilSearchMask.SetTo(new MCvScalar(0));
            CvInvoke.Circle(pupilSearchMask, new Point(config.PupilSearchRadius, config.PupilSearchRadius), config.PupilSearchRadius, new MCvScalar(255), -1, LineType.EightConnected, 0);
            // Test if this is error by not filling cuz of mask
            CvInvoke.BitwiseAnd(ImgPupilEdge, pupilSearchMask, ImgPupilEdgeMasked, null);
            
            var edgePts = new VectorOfPoint();
            CvInvoke.FindNonZero(ImgPupilEdgeMasked, edgePts);
            var pupilPtsRaw = new VectorOfPoint();
            if (edgePts.Size != 0)
            {
                CvInvoke.ConvexHull(edgePts, pupilPtsRaw, false);
                var pupilContour = Mat.Zeros(ImgPupilEdgeMasked.Rows, ImgPupilEdgeMasked.Cols, DepthType.Cv8U, 1);
                CvInvoke.DrawContours(pupilContour, new VectorOfVectorOfPoint(pupilPtsRaw), 0, new MCvScalar(255));
                CvInvoke.FindNonZero(pupilContour, pupilPtsRaw);
            } // TODO add error


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

            RotatedRect pupil;
            if (pupilPts.Size >= 5)
            {
                pupil = CvInvoke.FitEllipseDirect(pupilPts);
            } else
            {
                // TODO retrun error code here
                pupil = new RotatedRect(new PointF(0, 0), new SizeF(0, 0), 0);
            }
            output.Pupil = pupil;

            EyeTrackerDebug.TrackProcessingTime("Pupil Fit");

            // Mask out the exterior of the pupil
            var pupilMaskCenter = pupil.Center;
            pupilMaskCenter.X -= pupilSearchOffset.X;
            pupilMaskCenter.Y -= pupilSearchOffset.Y;

            var pupilMaskSize = pupil.Size;
            pupilMaskSize.Width *= config.PupilMaskPercent;
            pupilMaskSize.Height *= config.PupilMaskPercent;

            var pupilMaskEllipse = new RotatedRect(pupilMaskCenter, pupilMaskSize, pupil.Angle);

            var pupilMask = new Mat(pupilSearchRect.Size, DepthType.Cv8U, 1);
            pupilMask.SetTo(new MCvScalar(0));
            CvInvoke.Ellipse(pupilMask, pupilMaskEllipse, new MCvScalar(255), -1, LineType.EightConnected);

            var pupilROI = new Mat(ImgBlur, pupilSearchRect);

            ImgPupil.SetTo(new MCvScalar(0));
            CvInvoke.BitwiseAnd(pupilROI, pupilROI, ImgPupil, pupilMask);

            EyeTrackerDebug.TrackProcessingTime("Pupil Mask");
            // Find P1
            var imP1DsSize = new Size(pupilSearchRect.Width / config.P1DsFactor, pupilSearchRect.Height / config.P1DsFactor);
            CvInvoke.Resize(ImgPupil, ImgP1Ds, imP1DsSize, 0, 0, Inter.Linear);
            CvInvoke.Threshold(ImgP1Ds, ImgP1Thresh, config.P1Threshold, 255, ThresholdType.Binary);

            var p1COM = GetCenterOfMass(ImgP1Thresh);
            p1COM.X *= config.P1DsFactor;
            p1COM.Y *= config.P1DsFactor;

            output.P1Est = new Point((int)p1COM.X + pupilSearchOffset.X, (int)p1COM.Y + pupilSearchOffset.Y);
            if (EyeTracker.DEBUG)
            {
                ImgPupil.CopyTo(ImgPupilDebug);
            }
            var p1 = LocalizeSpot(ImgPupil, new Point((int)p1COM.X, (int)p1COM.Y), config.P1RoiRadius, config.PupilThreshold, true);
            output.P1 = new PointF(p1.X + pupilSearchOffset.X, p1.Y + pupilSearchOffset.Y);
            EyeTrackerDebug.TrackProcessingTime("Find P1");
            // Find P4
            Point maxLoc = new(), minLoc = new();
            double minVal = 0.0, maxVal = 0.0;
            CvInvoke.MinMaxLoc(ImgPupil, ref maxVal, ref minVal, ref minLoc, ref maxLoc);

            output.P4Est = new Point(maxLoc.X + pupilSearchOffset.X, maxLoc.Y + pupilSearchOffset.Y);

            var p4 = LocalizeSpot(ImgPupil, maxLoc, config.P4RoiRadius, config.P4Threshold, false);
            output.P4 = new PointF(p4.X + pupilSearchOffset.X, p4.Y + pupilSearchOffset.Y);

            EyeTrackerDebug.TrackProcessingTime("Find P4");
            if (EyeTracker.DEBUG)
            {
                var debugFulMat = DrawFullDebug(output, config);
                EyeTrackerDebug.AddImage("DPI-Overview", imageEye.WhichEye, debugFulMat.ToImage<Bgr, byte>());

                var debugP1P4Mat = DrawP1P4Debug(output, config);
                EyeTrackerDebug.AddImage("DPI-P1P4", imageEye.WhichEye, debugP1P4Mat.ToImage<Bgr, byte>());

                var debugPupil = DrawPupilDebug(output, config);
                EyeTrackerDebug.AddImage("DPI-Pupil", imageEye.WhichEye, debugPupil.ToImage<Bgr, byte>());
            }

            return output;
        }

        public void Dispose ()
        {

        }
    }
    
}
