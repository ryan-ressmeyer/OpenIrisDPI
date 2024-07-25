namespace OpenIris
{
#nullable enable

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
    using System.Runtime;
    using Emgu.CV.Cuda;
    using Emgu.CV.Ocl;
    using Emgu.CV.Util;
    using System.Diagnostics;

    /// <summary>
    /// Class in charge of processing images and tracking the pupil and iris to obtain the eye
    /// position and the torsion angle.
    /// </summary>
    [Export(typeof(EyeTrackingPipelineBase)), PluginDescriptionAttribute("DPI", typeof(EyeTrackingPipelineDPISettings))]
    public sealed class EyeTrackingPipelineDPI : EyeTrackingPipelineBase
    {
        private OpenIrisDPI dpi = new OpenIrisDPI();
        private OpenIrisDPI dpi_gui = new OpenIrisDPI();
        /// <summary>
        /// Disposes objects.
        /// </summary>
        public override void Dispose()
        {
            dpi.Dispose();

            base.Dispose();
        }

        private OpenIrisDPIConfig ConvertSettingsToDPIConfig(ImageEye imageEye, EyeTrackingPipelineDPISettings trackingSettings)
        {
            var dpiConfig = new OpenIrisDPIConfig();

            var eyeROI = imageEye.WhichEye == Eye.Left ? Settings.CroppingLeftEye : Settings.CroppingRightEye;
            dpiConfig.Crop = new Rectangle(
                new Point(eyeROI.Left, eyeROI.Top),
                new Size((imageEye.Size.Width - eyeROI.Left - eyeROI.Width), (imageEye.Size.Height - eyeROI.Top - eyeROI.Height)));

            dpiConfig.BlurRadius = imageEye.WhichEye == Eye.Left ? trackingSettings.BlurRadiusLeftEye : trackingSettings.BlurRadiusRightEye;

            dpiConfig.PupilThreshold = imageEye.WhichEye == Eye.Left ? trackingSettings.PupilThresholdLeftEye : trackingSettings.PupilThresholdRightEye;


            dpiConfig.PupilSearchRadius = imageEye.WhichEye == Eye.Left ? trackingSettings.PupilSearchRadiusLeftEye : trackingSettings.PupilSearchRadiusRightEye;

            float dsFactor = imageEye.WhichEye == Eye.Left ? trackingSettings.DsFactorLeftEye : trackingSettings.DsFactorRightEye;
            
            if (dsFactor <= 0.0)
            {
                dpiConfig.PupilDsFactor = 1;

                dpiConfig.PupilFitDsFactor = 1;

                dpiConfig.P1DsFactor = 1;
            } else
            {
                dpiConfig.PupilDsFactor = (int) Math.Ceiling(dpiConfig.PupilSearchRadius * .1 * dsFactor);

                dpiConfig.PupilFitDsFactor = (int) Math.Ceiling(dpiConfig.PupilSearchRadius * .02 * dsFactor);

                dpiConfig.P1DsFactor = (int) Math.Ceiling(dpiConfig.PupilSearchRadius * .01 * dsFactor);
            }

            dpiConfig.PupilMaskErodeRadius = imageEye.WhichEye == Eye.Left ? trackingSettings.PupilMaskErodeRadiusLeftEye : trackingSettings.PupilMaskErodeRadiusRightEye;


            dpiConfig.P1Threshold = imageEye.WhichEye == Eye.Left ? trackingSettings.P1ThresholdLeftEye : trackingSettings.P1ThresholdRightEye;

            dpiConfig.P1RoiRadius = imageEye.WhichEye == Eye.Left ? trackingSettings.P1RoiRadiusLeftEye : trackingSettings.P1RoiRadiusRightEye;


            dpiConfig.P4RoiRadius = imageEye.WhichEye == Eye.Left ? trackingSettings.P4RoiRadiusLeftEye : trackingSettings.P4RoiRadiusRightEye;

            dpiConfig.PupilAlgorithm = imageEye.WhichEye == Eye.Left ? trackingSettings.PupilAlgorithmLeftEye : trackingSettings.PupilAlgorithmRightEye;

            return dpiConfig;
        }


        /// <summary>
        /// Convert OpenIrisDPIOutput to EyeData.
        /// </summary>
        private EyeData ConvertDPIOutputToEyeData(OpenIrisDPIOutput output, ImageEye imageEye, OpenIrisDPIConfig dpiConfig)
        {
            var pupil = new PupilData(output.Pupil.Center + new Size(dpiConfig.Crop.Location), output.Pupil.Size, output.Pupil.Angle);

            // Store Various Estimators in the 
            CornealReflectionData[] crs =
            {
                new CornealReflectionData(output.P1.Center + new Size(dpiConfig.Crop.Location), output.P1.Size, output.P1.Angle),
                new CornealReflectionData(),
                new CornealReflectionData(),
                new CornealReflectionData(output.P4.Center + new Size(dpiConfig.Crop.Location), output.P4.Size, output.P4.Angle),
            };

            // Store Pupil Search Radius in the Iris
            var iris = new IrisData(output.PupilEst + new Size(dpiConfig.Crop.Location), dpiConfig.PupilSearchRadius);

            // Create the data structure
            var eyeData = new EyeData
            {
                WhichEye = imageEye.WhichEye,
                Timestamp = imageEye.TimeStamp,
                ImageSize = imageEye.Size,
                ProcessFrameResult = ProcessFrameResult.Good,
                Iris = iris,
                Pupil = pupil,
                CornealReflections = crs,
                TorsionAngle = 0.0,
                Eyelids = new EyelidData(),
                DataQuality = 100.0,
            };

            return eyeData;
        }

        
        /// <summary>
        /// Process an input ImageEye using a given set of settings using the DPI algorithm.
        /// </summary>
        /// <returns></returns>
        private EyeData ProcessImage(ImageEye imageEye, EyeTrackingPipelineDPISettings trackingSettings)
        {
            // Cropping rectangle and eye ROI (minimum size 20x20 pix)

            var dpiConfig = ConvertSettingsToDPIConfig(imageEye, trackingSettings); 

            var output = dpi.FindDualPurkinje(imageEye, dpiConfig);

            var eyeData = ConvertDPIOutputToEyeData(output, imageEye, dpiConfig);

            return eyeData;
        }

        /// <summary>
        /// Process images.
        /// </summary>
        /// <param name="imageEye"></param>
        /// <param name="eyeCalibrationParameters"></param>
        /// <returns></returns>
        public override (EyeData data, Image<Gray, byte>? imateTorsion) Process(ImageEye imageEye, EyeCalibration eyeCalibrationParameters)
        {
            var trackingSettings = Settings as EyeTrackingPipelineDPISettings ?? throw new Exception("Wrong type of settings");

            return (this.ProcessImage(imageEye, trackingSettings), null);
        }


        /// <summary>
        /// Draw the UI images.
        /// </summary>
        /// <returns></returns>
        public override IInputArray? UpdatePipelineEyeImage(Eye whichEye, EyeTrackerImagesAndData dataAndImages)
        {
            EyeTrackingPipelineDPISettings? settings = dataAndImages.TrackingSettings as EyeTrackingPipelineDPISettings;
            if (settings == null)
            {
                return null;
            }

            ImageEye? image = dataAndImages.Images[whichEye];
            if (image == null)
            {
                return null;
            }

            var dpiConfig = ConvertSettingsToDPIConfig(image, settings);

            var output = dpi_gui.FindDualPurkinje(image, dpiConfig);

            return dpi_gui.DrawFullDebug(output, dpiConfig);
        }

        /// <summary>
        /// Get the list of tracking settings that will be shown as sliders in the setup UI.
        /// </summary>
        /// <returns></returns>
        public override List<(string text, RangeDouble range, string settingName)>? GetQuickSettingsList()
        {
            var theSettings = Settings as EyeTrackingPipelineDPISettings ?? throw new InvalidOperationException("bad settings");

            var list = new List<(string text, RangeDouble range, string SettingName)>();


            var settingName = WhichEye switch
            {
                Eye.Left => nameof(theSettings.PupilThresholdLeftEye),
                Eye.Right => nameof(theSettings.PupilThresholdRightEye),
            };
            list.Add(("Pupil threshold", new RangeDouble(0, 255), settingName));

            settingName = WhichEye switch
            {
                Eye.Left => nameof(theSettings.PupilSearchRadiusLeftEye),
                Eye.Right => nameof(theSettings.PupilSearchRadiusRightEye),
            };
            list.Add(("Pupil Search Radius", new RangeDouble(0, 400), settingName));

            settingName = WhichEye switch
            {
                Eye.Left => nameof(theSettings.P1ThresholdLeftEye),
                Eye.Right => nameof(theSettings.P1ThresholdRightEye),
            };
            list.Add(("CR threshold", new RangeDouble(0, 255), settingName));

            settingName = WhichEye switch
            {
                Eye.Left => nameof(theSettings.P1RoiRadiusLeftEye),
                Eye.Right => nameof(theSettings.P1RoiRadiusRightEye),
            };
            list.Add(("CR ROI Radius", new RangeDouble(0, 100), settingName));

            settingName = WhichEye switch
            {
                Eye.Left => nameof(theSettings.P4RoiRadiusLeftEye),
                Eye.Right => nameof(theSettings.P4RoiRadiusRightEye),
            };
            list.Add(("P4 ROI Radius", new RangeDouble(0, 255), settingName));

            return list;
        }
    }

#pragma warning disable CS1591 // Missing XML comment for publicly visible type or member

    /// <summary>
    /// Settings for any pipeline that uses thresholds for pupil and reflections. Important to make them compatible with the remote
    /// UI client
    /// </summary>
    [Serializable]
    public class EyeTrackingPipelineDPISettings : EyeTrackingPipelineSettings
    {
        // Gaussian Blur
        [Category("General settings"), Description("Set the radius of gaussian blur to be applied to the image (left eye)")]
        public int BlurRadiusLeftEye { get => blurRadiusLeftEye; set => SetProperty(ref blurRadiusLeftEye, value, nameof(BlurRadiusLeftEye)); }
        private int blurRadiusLeftEye = 2;

        [Category("General settings"), Description("Set the radius of gaussian blur to be applied to the image (right eye)")]
        public int BlurRadiusRightEye { get => blurRadiusRightEye; set => SetProperty(ref blurRadiusRightEye, value, nameof(blurRadiusRightEye)); }
        private int blurRadiusRightEye = 2;

        // Pupil Threshold
        [Category("Pupil settings"), Description("Threshold to find the dark pixels that belong to the pupil (left eye).")]
        public int PupilThresholdLeftEye { get => pupilThresholdLeftEye; set => SetProperty(ref pupilThresholdLeftEye, value, nameof(PupilThresholdLeftEye)); }
        private int pupilThresholdLeftEye = 30;

        [Category("Pupil settings"), Description("Threshold to find the dark pixels that belong to the pupil (right eye).")]
        public int PupilThresholdRightEye { get => pupilThresholdRightEye; set => SetProperty(ref pupilThresholdRightEye, value, nameof(PupilThresholdRightEye)); }
        private int pupilThresholdRightEye = 30;

        // Pupil Search Radius
        [Category("Pupil settings"), Description("Search radius for the pupil boarder. Units of pixels (left eye).")]
        public int PupilSearchRadiusLeftEye { get => pupilSearchRadiusLeftEye; set => SetProperty(ref pupilSearchRadiusLeftEye, value, nameof(PupilSearchRadiusLeftEye)); }
        private int pupilSearchRadiusLeftEye = 200;

        [Category("Pupil settings"), Description("Search radius for the pupil boarder. Units of pixels (right eye).")]
        public int PupilSearchRadiusRightEye { get => pupilSearchRadiusRightEye; set => SetProperty(ref pupilSearchRadiusRightEye, value, nameof(PupilSearchRadiusRightEye)); }
        private int pupilSearchRadiusRightEye = 200;

        // Pupil Algorithm
        [Category("Pupil settings"), Description("Algorithm to use for pupil fitting (left eye).")]
        public PupilAlgorithm PupilAlgorithmLeftEye { get => pupilAlgorithmLeftEye; set => SetProperty(ref pupilAlgorithmLeftEye, value, nameof(PupilAlgorithmLeftEye)); }
        private PupilAlgorithm pupilAlgorithmLeftEye = PupilAlgorithm.Moments;

        [Category("Pupil settings"), Description("Algorithm to use for pupil fitting (left eye).")]
        public PupilAlgorithm PupilAlgorithmRightEye { get => pupilAlgorithmRightEye; set => SetProperty(ref pupilAlgorithmRightEye, value, nameof(PupilAlgorithmRightEye)); }
        private PupilAlgorithm pupilAlgorithmRightEye = PupilAlgorithm.Moments;

        // P1 Threshold
        [Category("CR settings"), Description("Threshold for the corneal reflection (1st Purkinje image) (left eye).")]
        public int P1ThresholdLeftEye { get => p1ThresholdLeftEye; set => SetProperty(ref p1ThresholdLeftEye, value, nameof(P1ThresholdLeftEye)); }
        private int p1ThresholdLeftEye = 245;

        [Category("CR settings"), Description("Threshold for the corneal reflection (1st Purkinje image) (right eye).")]
        public int P1ThresholdRightEye { get => p1ThresholdRightEye; set => SetProperty(ref p1ThresholdRightEye, value, nameof(P1ThresholdRightEye)); }
        private int p1ThresholdRightEye = 245;

        // P1 ROI 
        [Category("CR settings"), Description("Radius of the region of interest used for localizing the corneal reflection. Units of pixels (left eye).")]
        public int P1RoiRadiusLeftEye { get => p1RoiRadiusLeftEye; set => SetProperty(ref p1RoiRadiusLeftEye, value, nameof(P1RoiRadiusLeftEye)); }
        private int p1RoiRadiusLeftEye = 40;

        [Category("CR settings"), Description("Radius of the region of interest used for localizing the corneal reflection. Units of pixels (right eye).")]
        public int P1RoiRadiusRightEye { get => p1RoiRadiusRightEye; set => SetProperty(ref p1RoiRadiusRightEye, value, nameof(P1RoiRadiusRightEye)); }
        private int p1RoiRadiusRightEye = 40;

        // Pupil Mask Percent
        [Category("P4 settings"), Description("Radius of erosion of pupil used to mask out everything outside of the pupil (left eye).")]
        public int PupilMaskErodeRadiusLeftEye { get => pupilMaskErodeRadiusLeftEye; set => SetProperty(ref pupilMaskErodeRadiusLeftEye, value, nameof(PupilMaskErodeRadiusLeftEye)); }
        private int pupilMaskErodeRadiusLeftEye = 3;

        [Category("P4 settings"), Description("Radius of erosion of pupil used to mask out everything outside of the pupil (right eye).")]
        public int PupilMaskErodeRadiusRightEye { get => pupilMaskErodeRadiusRightEye; set => SetProperty(ref pupilMaskErodeRadiusRightEye, value, nameof(PupilMaskErodeRadiusRightEye)); }
        private int pupilMaskErodeRadiusRightEye = 3;

        // P4 ROI
        [Category("P4 settings"), Description("Radius of the region of interest used for localizing P4. Units of pixels (left eye).")]
        public int P4RoiRadiusLeftEye { get => p4RoiRadiusLeftEye; set => SetProperty(ref p4RoiRadiusLeftEye, value, nameof(P4RoiRadiusLeftEye)); }
        private int p4RoiRadiusLeftEye = 15;

        [Category("P4 settings"), Description("Radius of the region of interest used for localizing P4. Units of pixels (right eye).")]
        public int P4RoiRadiusRightEye { get => p4RoiRadiusRightEye; set => SetProperty(ref p4RoiRadiusRightEye, value, nameof(P4RoiRadiusRightEye)); }
        private int p4RoiRadiusRightEye = 15;

        // Pupil Downsampling Factor
        [Category("Performance Settings"), Description("Downsampling factor (left eye).")]
        public float DsFactorLeftEye { get => dsFactorLeftEye; set => SetProperty(ref dsFactorLeftEye, value, nameof(DsFactorLeftEye)); }
        private float dsFactorLeftEye = 1.0f;

        [Category("Performance Settings"), Description("Downsampling factor (right eye).")]
        public float DsFactorRightEye { get => dsFactorRightEye; set => SetProperty(ref dsFactorRightEye, value, nameof(DsFactorRightEye)); }
        private float dsFactorRightEye = 1.0f;
    }
}
