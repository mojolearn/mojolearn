// Returns the lowest index in `array` whose value is greater or equal to `element`.
// Values outside the quantile range are clamped to the edge bins: values below the
// first quantile return 0, and values above the last quantile return len - 1.
template <typename DataT, typename IdxT>
HDI IdxT lower_bound(DataT const* array, IdxT len, DataT element)
{
  IdxT start = 0;
  IdxT end   = len - 1;
  IdxT mid;
  while (start < end) {
    mid = (start + end) / 2;
    if (array[mid] < element) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}
