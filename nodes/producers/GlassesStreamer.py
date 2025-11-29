############
#
# Copyright (c) 2024 Maxim Yudayev and KU Leuven eMedia Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Created 2024-2025 for the KU Leuven AidWear, AidFOG, and RevalExo projects
# by Maxim Yudayev [https://yudayev.com].
#
# ############

import time
from typing import Callable
import uvc
from multiprocessing import Process, Queue, Event
from queue import Empty

from nodes.producers.Producer import Producer
from streams import GlassesStream

from utils.mp_utils import launch_callable
from utils.time_utils import get_time, init_time
from utils.zmq_utils import PORT_BACKEND, PORT_KILL, PORT_SYNC_HOST


#######################################################
#######################################################
# A class for streaming videos from Pupil core cameras.
#######################################################
#######################################################
class GlassesHandler:
  def __call__(self,
               ref_time_s: float,
               camera_name: str,
               camera_spec: dict,
               queue: Queue,
               video_image_format: str,
               stop_event: Event,
               keep_event: Event,
               ready_event: Event
  ):
    init_time(ref_time_s)
    self.queue = queue
    self.camera_name = camera_name
    self.camera_spec = camera_spec
    self.cap: uvc.Capture

    self._restart_cap_object()

    if video_image_format == "mjpeg":
      get_buffer_fn = lambda frame: bytes(frame.jpeg_buffer)
    elif video_image_format == "bgr":
      get_buffer_fn = lambda frame: frame.bgr
    elif video_image_format == "yuv":
      get_buffer_fn = lambda frame: frame.yuv
    else:
      get_buffer_fn = lambda _: None

    ready_event.set()
    keep_event.wait()

    while not stop_event.is_set():
      self._get_frame(get_buffer_fn)
    self.cap.close()


  def _restart_cap_object(self) -> None:
    try:
      devices = dict(map(lambda dev: (dev['name'], dev['uid']), uvc.device_list()))

      self.cap = uvc.Capture(devices[self.camera_spec['name']])
      self.cap.bandwidth_factor = self.camera_spec['bandwidth_factor']

      for mode in self.cap.available_modes:
        if (mode.width == self.camera_spec['resolution'][1] and
            mode.height == self.camera_spec['resolution'][0] and
            mode.fps == self.camera_spec['fps']):
          self.cap.frame_mode = mode
          break
        # configure the controls on each `Capture` object (exposure, brightness, sharpness, etc)
        controls_by_name = {c.display_name: c for c in self.cap.controls}

      print(f"Settings controls for {self.camera_spec['name']}", flush=True) 
      for ctrl_name, value in self.camera_spec.get('uvc_controls', {}).items():
        ctrl = controls_by_name.get(ctrl_name)
        try:
          ctrl.value = value
        except Exception as e:
          print(f"Could not set control for {self.camera_spec['name']} '{ctrl_name}' to {value}: {e}")
    except:
      time.sleep(1)


  def _get_frame(self, get_buffer_fn: Callable) -> None:
    try:
      frame = self.cap.get_frame(timeout=1)
      toa_s = get_time()
      out = {
        "timestamp": frame.timestamp,
        "index": frame.index,
        "data": get_buffer_fn(frame)
      }
      self.queue.put((self.camera_name, out, toa_s))
    except uvc.InitError as err:
      print(f"[GlassesStreamer] Failed to init {self.camera_name}: {err}", flush=True)
    except uvc.StreamError as err:
      print(f"[GlassesStreamer] Stream error for {self.camera_name}: {err}", flush=True)
    except (TimeoutError, NameError, AttributeError) as err:
      print(f"[GlassesStreamer] Reconnecting {self.camera_name}: {err}", flush=True)
      self._restart_cap_object()


class GlassesStreamer(Producer):
  @classmethod
  def _log_source_tag(cls) -> str:
    return 'glasses'


  def __init__(self,
               host_ip: str,
               camera_mapping: dict,
               logging_spec: dict,
               video_image_format: str = "mjpeg", # [bgr, mjpeg, yuv]
               port_pub: str = PORT_BACKEND,
               port_sync: str = PORT_SYNC_HOST,
               port_killsig: str = PORT_KILL,
               transmit_delay_sample_period_s: float = float('nan'),
               timesteps_before_solidified: int = 0,
               **_):

    self._camera_mapping = camera_mapping
    self._video_image_format = video_image_format
    self._is_continue_grabbing = True
    self._start_index: dict[str, int | None] = dict(map(lambda cam: (cam, None), self._camera_mapping.keys()))
    self._parse_frame_fn = self._parse_first_frame
    self._cap_queue: Queue = Queue()
    self._stop_event: Event = Event()
    self._keep_event: Event = Event()
    self._ref_time_s = logging_spec['ref_time_s']
    
    stream_info = {
      "camera_mapping": self._camera_mapping,
      "pixel_format": video_image_format,
      "timesteps_before_solidified": timesteps_before_solidified
    }

    super().__init__(host_ip=host_ip,
                     stream_info=stream_info,
                     logging_spec=logging_spec,
                     port_pub=port_pub,
                     port_sync=port_sync,
                     port_killsig=port_killsig,
                     transmit_delay_sample_period_s=transmit_delay_sample_period_s)


  @classmethod
  def create_stream(cls, stream_info: dict) -> GlassesStream:
    return GlassesStream(**stream_info)


  def _ping_device(self) -> None:
    return None


  def _connect(self) -> bool:
    self._cap_procs: list[Process] = []
    self._cap_handlers: list[GlassesHandler] = []
    ready_events: list[Event] = []
    # launch each capture subprocess
    for cam in self._camera_mapping.keys():
      handler = GlassesHandler()
      ready_event = Event()
      proc = Process(target=launch_callable, args=(handler,
                                                   self._ref_time_s,
                                                   cam,
                                                   self._camera_mapping[cam],
                                                   self._cap_queue,
                                                   self._video_image_format,
                                                   self._stop_event,
                                                   self._keep_event,
                                                   ready_event))
      self._cap_procs.append(proc)
      self._cap_handlers.append(handler)
      ready_events.append(ready_event)
      proc.start()
    # Will wait until child UVC processes have set up
    for event in ready_events: event.wait()
    return True


  def _keep_samples(self) -> None:
    self._keep_event.set()


  def _process_data(self) -> None:
    try:
      msg = self._cap_queue.get(timeout=10)
    except Empty:
      if not self._is_continue_capture:
        self._send_end_packet()
      return

    process_time_s = get_time()
    output = self._parse_frame_fn(msg)
    if output is None:
      return
    tag: str = "%s.data" % self._log_source_tag()
    self._publish(tag, process_time_s=process_time_s, data=output)


  def _parse_first_frame(self, msg: tuple) -> dict:
    camera_name, frame, toa_s = msg

    if self._start_index[camera_name] is None:
      self._start_index[camera_name] = frame['index']
      frame_index = 0
    else:
      frame_index = frame['index'] - self._start_index[camera_name]

    if all(v is not None for v in self._start_index.values()):
      self._parse_frame_fn = self._parse_frame

    output: dict[str, dict] = {}
    output[camera_name] = {
      'frame_timestamp': frame['timestamp'],
      'frame_index': frame_index,
      'frame_sequence_id': frame['index'],
      'frame': (frame['data'], False, frame_index),
      'toa_s': toa_s
    }
    return output

  def _parse_frame(self, msg: tuple) -> dict:
    camera_name, frame, toa_s = msg

    frame_index = frame['index'] - self._start_index[camera_name]

    output: dict[str, dict] = {}
    output[camera_name] = {
      'frame_timestamp': frame['timestamp'],
      'frame_index': frame_index,
      'frame_sequence_id': frame['index'],
      'frame': (frame['data'], False, frame_index),
      'toa_s': toa_s
    }
    return output


  def _stop_new_data(self) -> None:
    self._stop_event.set()


  def _cleanup(self) -> None:
    for proc in self._cap_procs: proc.join()
    super()._cleanup()
