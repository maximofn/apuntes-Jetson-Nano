import argparse
import sys
sys.path.append('/opt/nvidia/deepstream/deepstream-6.1/deepstream_python_apps/apps/')
import gi # PyGObject is a Python package that enables links to librarys based on GObjects like GTK , GStreamer , WebKitGTK , GLib , GIO and more.
gi.require_version('Gst', '1.0') # GStreamer
gi.require_version('GstRtspServer', '1.0') # GStreamer rtsp server
from gi.repository import GObject, Gst, GstRtspServer, GLib
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS
import pyds

fps_streams = {}
getfps_streams = {}
ready = False

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def parse_args():
    parser = argparse.ArgumentParser(description='RTSP Output Sample Application Help ')
    parser.add_argument("-i", "--input-video", help="Path to v4l2 device path such as /dev/video0", required=True)
    parser.add_argument("-o", "--output", default=None, help="Set the output file path ")
    parser.add_argument("--input-codec", default="H264", help="Input Codec H264/H265, default=H264", choices=['H264','H265'])
    parser.add_argument("--output-codec", default="H264", help="RTSP Streaming Codec H264/H265, default=H264", choices=['H264','H265'])
    parser.add_argument("-b", "--bitrate", default=4000000, help="Set the encoding bitrate, default=4000000", type=int)
    parser.add_argument("-p", "--port", default=8554, help="Port of RTSP stream, default=8554", type=int)
    parser.add_argument("-s", "--stream-name", default="stream1", help="Stream name, default=stream1")
    
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    
    global stream_path
    global output_path
    global input_codec
    global output_codec
    global bitrate
    global port
    global stream_name
    
    stream_path = args.input_video
    output_path = args.output
    input_codec = args.input_codec
    output_codec = args.output_codec
    bitrate = args.bitrate
    port = args.port
    stream_name = args.stream_name
    
    return 0


def osd_sink_pad_buffer_probe(pad,info,u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return
    print(type(pad))
    
    global ready
    if ready == False:
        ready = True
        print("\n Ready to stream")
    
    fps_streams[0].update_fps()
    fps = fps_streams[0].get_fps()

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break

        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        py_nvosd_text_params.display_text = f"Frame Number={frame_number} fps={fps}"
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 20
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        py_nvosd_text_params.set_bg_clr = 1
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.5)
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
        
    return Gst.PadProbeReturn.OK


def main(args):
    if len(args) < 2:
        sys.stderr.write("usage: %s -i <uri1> [uri2] ... [uriN]\n" % args[0])
        sys.exit(1)

    fps_streams[0] = GETFPS(0)
    number_sources=len(args)-2

    gst_status, _ = Gst.init_check(None)    # GStreamer initialization
    if not gst_status:
        sys.stderr.write("Unable to initialize Gst\n")
        sys.exit(1)
    
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
    if not source:
        sys.stderr.write(" Unable to create source \n")
    caps_v4l2src = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
    if not caps_v4l2src:
        sys.stderr.write(" Unable to create v4l2src capsfilter \n")
    vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
    if not vidconvsrc:
        sys.stderr.write(" Unable to create videoconvert \n")
    nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
    if not nvvidconvsrc:
        sys.stderr.write(" Unable to create nvvideoconvert \n")
    caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    if not caps_vidconvsrc:
        sys.stderr.write(" Unable to create capsfilter \n")
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
    if not nvvidconv_postosd:
        sys.stderr.write(" Unable to create nvvidconv_postosd \n")
    caps = Gst.ElementFactory.make("capsfilter", "filter")
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
    if output_codec == "H264":
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
    elif output_codec == "H265":
        encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
    if not encoder:
        sys.stderr.write(" Unable to create encoder")
    encoder.set_property('bitrate', bitrate)
    if is_aarch64():
        encoder.set_property('preset-level', 1)
        encoder.set_property('insert-sps-pps', 1)
        encoder.set_property('bufapi-version', 1)
    if output_codec == "H264":
        rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
    elif output_codec == "H265":
        rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")
    if not rtppay:
        sys.stderr.write(" Unable to create rtppay")
    updsink_port_num = 5400
    sink = Gst.ElementFactory.make("udpsink", "udpsink")
    if not sink:
        sys.stderr.write(" Unable to create udpsink")
        
    if output_path is None:
        sink.set_property('host', '224.224.255.255')
        sink.set_property('port', updsink_port_num)
        sink.set_property('async', False)
        sink.set_property('sync', 1)
    sink.set_property("qos",0)
    caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=30/1"))
    caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM)"))
    source.set_property('device', stream_path)
    streammux.set_property('live-source', 1)
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', 4000000)
    
    pipeline.add(source)
    pipeline.add(caps_v4l2src)
    pipeline.add(vidconvsrc)
    pipeline.add(nvvidconvsrc)
    pipeline.add(caps_vidconvsrc)
    pipeline.add(streammux)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps)
    pipeline.add(encoder)
    pipeline.add(sink)
    pipeline.add(rtppay)

    srcpad = caps_vidconvsrc.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of decoder \n")
    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    source.link(caps_v4l2src) # source-capsfilter
    caps_v4l2src.link(vidconvsrc) # capsfilter-videoconvert
    vidconvsrc.link(nvvidconvsrc) # videoconvert-nvvideoconvert
    nvvidconvsrc.link(caps_vidconvsrc) # nvvideoconvert-capsfilter
    srcpad.link(sinkpad) # capsfilter-streammux
    streammux.link(nvvidconv) # streammux-nvvidconv
    nvvidconv.link(nvosd) # nvvidconv-nvosd
    nvosd.link(nvvidconv_postosd) # nvosd-nvvidconv_postosd
    nvvidconv_postosd.link(caps) # nvvidconv_postosd-caps
    caps.link(encoder) # caps-encoder
    encoder.link(rtppay) # encoder-rtppay
    rtppay.link(sink) # rtppay-sink
    
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)
    
    server = GstRtspServer.RTSPServer.new()
    server.props.service = f"{port}"
    server.attach(None)
    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch( f"( udpsrc name=pay0 port={updsink_port_num} buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string){output_codec}, payload=96 \" )")
    factory.set_shared(True)
    server.get_mount_points().add_factory(f"/{stream_name}", factory)
    print("\t Launched RTSP Streaming at " + color.UNDERLINE + color.GREEN + f"rtsp://localhost:{port}/{stream_name}" + color.END)
    
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    parse_args()
    sys.exit(main(sys.argv))