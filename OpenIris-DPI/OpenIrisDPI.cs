using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OpenIris
{
#nullable enable

    using Emgu.CV;
    using Emgu.CV.Structure;
    using OpenIris.UI;
    using System;
    using System.Collections.Generic;
    using System.ComponentModel.Composition;

    /// <summary>
    /// Class in charge of processing images and tracking the pupil and the 1st and 4th Purkinje reflections.
    /// </summary>
    [Export(typeof(EyeTrackingPipelineBase)), PluginDescriptionAttribute("DPI", typeof(EyeTrackingPipelineSettings))]
    public sealed class EyeTrackingPipelineDPI : EyeTrackingPipelineBase, IDisposable
    {
        /// <summary>
        /// Process images.
        /// </summary>
        /// <param name="imageEye"></param>
        /// <param name="eyeCalibrationParameters"></param>
        /// <returns></returns>
        public override (EyeData data, Image<Gray, byte> imateTorsion) Process(ImageEye imageEye, EyeCalibration eyeCalibrationParameters)
        {
            return (new EyeData()
            {
                WhichEye = imageEye.WhichEye,
                Timestamp = imageEye.TimeStamp,
                ImageSize = imageEye.Size,
                ProcessFrameResult = ProcessFrameResult.Good,
            },
            new Image<Gray, byte>(0, 0));
        }
    }

}
