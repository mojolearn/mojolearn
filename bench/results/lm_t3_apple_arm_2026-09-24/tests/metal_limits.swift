import Metal
import Foundation
let d = MTLCreateSystemDefaultDevice()!
let o: [String: Any] = [
  "name": d.name,
  "recommendedMaxWorkingSetSize": d.recommendedMaxWorkingSetSize,
  "maxBufferLength": d.maxBufferLength,
  "hasUnifiedMemory": d.hasUnifiedMemory,
  "currentAllocatedSize": d.currentAllocatedSize,
  "physicalMemory": ProcessInfo.processInfo.physicalMemory,
  "os": ProcessInfo.processInfo.operatingSystemVersionString,
]
let j = try! JSONSerialization.data(withJSONObject: o, options: [.sortedKeys])
print(String(data: j, encoding: .utf8)!)
