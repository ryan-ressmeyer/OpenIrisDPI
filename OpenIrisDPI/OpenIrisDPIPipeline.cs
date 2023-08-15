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

    /// <summary>
    /// Class in charge of processing images and tracking the pupil and iris to obtain the eye
    /// position and the torsion angle.
    /// </summary>
    [Export(typeof(EyeTrackingPipelineBase)), PluginDescriptionAttribute("DPI", typeof(EyeTrackingPipelineDPISettings))]
    public sealed class EyeTrackingPipelineDPI : EyeTrackingPipelineBase
    {
        private OpenIrisDPI dpi = new OpenIrisDPI();

        /// <summary>
        /// Disposes objects.
        /// </summary>
        public override void Dispose()
        {
            dpi.Dispose();

            base.Dispose();
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

            // Cropping rectangle and eye ROI (minimum size 20x20 pix)

            var dpiConfig = new OpenIrisDPIConfig();

            var eyeROI = imageEye.WhichEye == Eye.Left ? Settings.CroppingLeftEye : Settings.CroppingRightEye;
            dpiConfig.Crop = new Rectangle(
                new Point(eyeROI.Left, eyeROI.Top),
                new Size((imageEye.Size.Width - eyeROI.Left - eyeROI.Width), (imageEye.Size.Height - eyeROI.Top - eyeROI.Height)));

            dpiConfig.BlurSize = imageEye.WhichEye == Eye.Left ? trackingSettings.BlurSizeLeftEye : trackingSettings.BlurSizeRightEye;

            dpiConfig.PupilThreshold = imageEye.WhichEye == Eye.Left ? trackingSettings.PupilThresholdLeftEye : trackingSettings.PupilThresholdRightEye;

            dpiConfig.PupilDsFactor = imageEye.WhichEye == Eye.Left ? trackingSettings.PupilDsFactorLeftEye : trackingSettings.PupilDsFactorRightEye;

            dpiConfig.PupilSearchRadius = imageEye.WhichEye == Eye.Left ? trackingSettings.PupilSearchRadiusLeftEye : trackingSettings.PupilSearchRadiusRightEye;

            dpiConfig.PupilMaskPercent = imageEye.WhichEye == Eye.Left ? trackingSettings.PupilMaskPercentLeftEye : trackingSettings.PupilMaskPercentRightEye;

            dpiConfig.P1DsFactor = imageEye.WhichEye == Eye.Left ? trackingSettings.P1DsFactorLeftEye : trackingSettings.P1DsFactorRightEye;

            dpiConfig.P1Threshold = imageEye.WhichEye == Eye.Left ? trackingSettings.P1ThresholdLeftEye : trackingSettings.P1ThresholdRightEye;

            dpiConfig.P1RoiRadius = imageEye.WhichEye == Eye.Left ? trackingSettings.P1RoiRadiusLeftEye : trackingSettings.P1RoiRadiusRightEye;

            dpiConfig.P4Threshold = imageEye.WhichEye == Eye.Left ? trackingSettings.P4ThresholdLeftEye : trackingSettings.P4ThresholdRightEye;

            dpiConfig.P4RoiRadius = imageEye.WhichEye == Eye.Left ? trackingSettings.P4RoiRadiusLeftEye : trackingSettings.P4RoiRadiusRightEye;

            var output = dpi.FindDualPurkinje(imageEye, dpiConfig);


            var pupil = new PupilData(output.Pupil.Center + new Size(dpiConfig.Crop.Location), output.Pupil.Size, output.Pupil.Angle);

            CornealReflectionData[] crs =
            {
                new CornealReflectionData(output.P1 + new Size(dpiConfig.Crop.Location), new SizeF(1.0f,1.0f), 0.0f),
                new CornealReflectionData(output.P1Fine + new Size(dpiConfig.Crop.Location), new SizeF(1.0f,1.0f), 0.0f),
                new CornealReflectionData(output.P1Est + new Size(dpiConfig.Crop.Location), new SizeF(1.0f,1.0f), 0.0f),
                new CornealReflectionData(output.P4 + new Size(dpiConfig.Crop.Location), new SizeF(1.0f,1.0f), 0.0f),
            };

            // Create the data structure
            var eyeData = new EyeData()
            {
                WhichEye = imageEye.WhichEye,
                Timestamp = imageEye.TimeStamp,
                ImageSize = imageEye.Size,
                ProcessFrameResult = ProcessFrameResult.Good,

                Pupil = pupil,
                CornealReflections = crs,
                TorsionAngle = 0.0,
                Eyelids = new EyelidData(),
                DataQuality = 100.0,
            };

            return (eyeData, null);
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
                Eye.Left => nameof(theSettings.P4ThresholdLeftEye),
                Eye.Right => nameof(theSettings.P4ThresholdRightEye),
            };
            list.Add(("P4 Threshold", new RangeDouble(0, 255), settingName));

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
        public int BlurSizeLeftEye { get => blurSizeLeftEye; set => SetProperty(ref blurSizeLeftEye, value, nameof(BlurSizeLeftEye)); }
        private int blurSizeLeftEye = 5;

        [Category("General settings"), Description("Set the radius of gaussian blur to be applied to the image (right eye)")]
        public int BlurSizeRightEye { get => blurSizeLeftEye; set => SetProperty(ref blurSizeRightEye, value, nameof(blurSizeRightEye)); }
        private int blurSizeRightEye = 5;

        // Pupil Threshold
        [Category("Pupil settings"), Description("Threshold to find the dark pixels that belong to the pupil (left eye).")]
        public int PupilThresholdLeftEye { get => pupilThresholdLeftEye; set => SetProperty(ref pupilThresholdLeftEye, value, nameof(PupilThresholdLeftEye)); }
        private int pupilThresholdLeftEye = 60;

        [Category("Pupil settings"), Description("Threshold to find the dark pixels that belong to the pupil (right eye).")]
        public int PupilThresholdRightEye { get => pupilThresholdRightEye; set => SetProperty(ref pupilThresholdRightEye, value, nameof(PupilThresholdRightEye)); }
        private int pupilThresholdRightEye = 60;

        // Pupil Downsampling Factor
        [Category("Pupil settings"), Description("Downsampling factor to find the pupil center of mass (left eye).")]
        public int PupilDsFactorLeftEye { get => pupilDsFactorLeftEye; set => SetProperty(ref pupilDsFactorLeftEye, value, nameof(PupilDsFactorLeftEye)); }
        private int pupilDsFactorLeftEye = 16;

        [Category("Pupil settings"), Description("Downsampling factor to find the pupil center of mass (right eye).")]
        public int PupilDsFactorRightEye { get => pupilDsFactorRightEye; set => SetProperty(ref pupilDsFactorRightEye, value, nameof(PupilDsFactorRightEye)); }
        private int pupilDsFactorRightEye = 16;

        // Pupil Search Radius
        [Category("Pupil settings"), Description("Search radius for the pupil boarder. Units of pixels (left eye).")]
        public int PupilSearchRadiusLeftEye { get => pupilSearchRadiusLeftEye; set => SetProperty(ref pupilSearchRadiusLeftEye, value, nameof(PupilSearchRadiusLeftEye)); }
        private int pupilSearchRadiusLeftEye = 60;

        [Category("Pupil settings"), Description("Search radius for the pupil boarder. Units of pixels (right eye).")]
        public int PupilSearchRadiusRightEye { get => pupilSearchRadiusRightEye; set => SetProperty(ref pupilSearchRadiusRightEye, value, nameof(PupilSearchRadiusRightEye)); }
        private int pupilSearchRadiusRightEye = 60;

        // Pupil Mask Percent
        [Category("Pupil settings"), Description("Percent of the pupil to use to mask out everything outside of the pupil (left eye).")]
        public float PupilMaskPercentLeftEye { get => pupilMaskPercentLeftEye; set => SetProperty(ref pupilMaskPercentLeftEye, value, nameof(PupilMaskPercentLeftEye)); }
        private float pupilMaskPercentLeftEye = 0.9f;

        [Category("Pupil settings"), Description("Percent of the pupil to use to mask out everything outside of the pupil (right eye).")]
        public float PupilMaskPercentRightEye { get => pupilMaskPercentRightEye; set => SetProperty(ref pupilMaskPercentRightEye, value, nameof(PupilMaskPercentRightEye)); }
        private float pupilMaskPercentRightEye = 0.9f;

        // P1 Threshold
        [Category("CR settings"), Description("Threshold for the corneal reflection (1st Purkinje image) (left eye).")]
        public int P1ThresholdLeftEye { get => p1ThresholdLeftEye; set => SetProperty(ref p1ThresholdLeftEye, value, nameof(P1ThresholdLeftEye)); }
        private int p1ThresholdLeftEye = 250;

        [Category("CR settings"), Description("Threshold for the corneal reflection (1st Purkinje image) (right eye).")]
        public int P1ThresholdRightEye { get => p1ThresholdRightEye; set => SetProperty(ref p1ThresholdRightEye, value, nameof(P1ThresholdRightEye)); }
        private int p1ThresholdRightEye = 250;

        // P1 Downsample Ratio
        [Category("CR settings"), Description("Downsampling ratio for finding the approximate CR center (left eye).")]
        public int P1DsFactorLeftEye { get => p1DsFactorLeftEye; set => SetProperty(ref p1DsFactorLeftEye, value, nameof(P1DsFactorLeftEye)); }
        private int p1DsFactorLeftEye = 2;

        [Category("CR settings"), Description("Downsampling ratio for finding the approximate CR center (right eye).")]
        public int P1DsFactorRightEye { get => p1DsFactorRightEye; set => SetProperty(ref p1DsFactorRightEye, value, nameof(P1DsFactorRightEye)); }
        private int p1DsFactorRightEye = 2;

        // P1 ROI 
        [Category("CR settings"), Description("Radius of the region of interest used for localizing the corneal reflection. Units of pixels (left eye).")]
        public int P1RoiRadiusLeftEye { get => p1RoiRadiusLeftEye; set => SetProperty(ref p1RoiRadiusLeftEye, value, nameof(P1RoiRadiusLeftEye)); }
        private int p1RoiRadiusLeftEye = 30;

        [Category("CR settings"), Description("Radius of the region of interest used for localizing the corneal reflection. Units of pixels (right eye).")]
        public int P1RoiRadiusRightEye { get => p1RoiRadiusRightEye; set => SetProperty(ref p1RoiRadiusRightEye, value, nameof(P1RoiRadiusRightEye)); }
        private int p1RoiRadiusRightEye = 30;

        // P4 ROI
        [Category("P4 settings"), Description("Threshold for localizing P4. Set just above pupil dark level (left eye).")]
        public int P4ThresholdLeftEye { get => p4ThresholdLeftEye; set => SetProperty(ref p4ThresholdLeftEye, value, nameof(P4ThresholdLeftEye)); }
        private int p4ThresholdLeftEye = 20;

        [Category("P4 settings"), Description("Threshold for localizing P4. Set just above pupil dark level (right eye).")]
        public int P4ThresholdRightEye { get => p4ThresholdRightEye; set => SetProperty(ref p4ThresholdRightEye, value, nameof(P4ThresholdRightEye)); }
        private int p4ThresholdRightEye = 20;

        // P4 ROI
        [Category("P4 settings"), Description("Radius of the region of interest used for localizing P4. Units of pixels (left eye).")]
        public int P4RoiRadiusLeftEye { get => p4RoiRadiusLeftEye; set => SetProperty(ref p4RoiRadiusLeftEye, value, nameof(P4RoiRadiusLeftEye)); }
        private int p4RoiRadiusLeftEye = 250;

        [Category("P4 settings"), Description("Radius of the region of interest used for localizing P4. Units of pixels (right eye).")]
        public int P4RoiRadiusRightEye { get => p4RoiRadiusRightEye; set => SetProperty(ref p4RoiRadiusRightEye, value, nameof(P4RoiRadiusRightEye)); }
        private int p4RoiRadiusRightEye = 250;

    }
}
