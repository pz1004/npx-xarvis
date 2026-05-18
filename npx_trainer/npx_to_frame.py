import numpy as np
import torch

def get_slice_metadata(data: torch.Tensor, bin_count: int, overlap: float=0.0):
    events = data
    times = events[:,3]
    time_window = (times[-1] - times[0]) // bin_count * (1 + overlap)
    stride = time_window * (1 - overlap)
    window_start_times = torch.arange(bin_count) * stride + times[0]
    window_end_times = window_start_times + time_window
    indices_start = torch.searchsorted(times, window_start_times)
    indices_end = torch.searchsorted(times, window_end_times)
    return list(zip(indices_start, indices_end))

#def slice_with_metadata(data: torch.Tensor, metadata: list[tuple[torch.Tensor, torch.Tensor]]):
def slice_with_metadata(data: torch.Tensor, metadata: list):
    return [data[start:end] for start, end in metadata]

def slice(data: torch.Tensor, bin_count: int, overlap: float=0.0):
    metadata = get_slice_metadata(data, bin_count, overlap)
    return slice_with_metadata(data, metadata)

def slice_events_by_time_bins(events, bin_count: int, overlap: float=0.0):
    return slice(events, bin_count, overlap)

# events: ndarray of shape [num_events, num_event_channels]
# sensor_size: size of the sensor that was used [W,H,P]
def npx_to_frame(events, sensor_size, n_time_bins: int, overlap: float=0.0):
  batch_size = 1
  event_slices = slice_events_by_time_bins(events, n_time_bins)
  frames = np.zeros((len(event_slices), batch_size, *sensor_size[::-1]))
  for i, event_slice in enumerate(event_slices):
    np.add.at(
      frames,
      (i, 0, event_slice[:,2], event_slice[:,1], event_slice[:,0]),
      1,
    )
  return torch.Tensor(frames)
