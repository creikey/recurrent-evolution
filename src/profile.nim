import times, deques, tables, options, math

const maxSamples: int = 300

type 
  ExecutingBlock = object
    startTime: float
  TimeResult = object
    average: float
    stddev: float

var samples = initTable[string, Deque[float]]()
var executingStack = newSeq[ExecutingBlock]()
var lastTotalTime = none[float]()

proc profileStart*(name: string) = 
  executingStack.add(ExecutingBlock(
    startTime: cpuTime(),
  ))
  if not samples.contains(name):
    samples[name] = initDeque[float](maxSamples)

proc profileEnd*(name: string) =
  var totalTime = cpuTime() - executingStack.pop().startTime
  if lastTotalTime.isSome():
    let newLastTotalTime = some[float](totalTime)
    totalTime -= lastTotalTime.get()
    lastTotalTime = newLastTotalTime
  else:
    lastTotalTime = some[float](totalTime)
  
  if executingStack.len == 0:
    lastTotalTime = none[float]()

  samples[name].addLast(totalTime)

template profile*(name: string, body: untyped) =
  profileStart(name)

  body

  profileEnd(name)
  

proc results*(): Table[string, TimeResult] =
  result = initTable[string, TimeResult]()
  for s in samples.keys():
    var sum: float = 0.0
    for time in samples[s]:
      sum += time
    let mean = sum / samples[s].len.float

    var variance: float = 0.0
    for time in samples[s]:
      variance += (time - mean) ^ 2
    variance /= (samples[s].len.float - 1)
    let stddev = sqrt(variance)

    result[s] = TimeResult(
      average: mean,
      stddev: stddev
    )

proc printresults*() =
  echo "Profiled block | Average | Standard Deviation"
  let r = results()
  for s in r.keys():
    echo s, " | ", r[s].average, " | ", r[s].stddev

when isMainModule:
  import os
  profile "outer block":
    sleep(500)
    profile "inner block":
      sleep(500)
  profile "second outer block":
    sleep(500)
    profile "other inner":
      sleep(100)
  printresults()