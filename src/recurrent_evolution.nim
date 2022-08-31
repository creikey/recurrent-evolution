import arraymancer, print, chroma, vmath, times, chroma/transformations, algorithm, flatty, os, options

import profile


const map_color = color(0.5, 0.5, 0.5)
const creature_start_color = color(1, 1, 1)
const map_size: float = 200.0
const init_creatures = 50
const creature_radius: float = 0.5
const baby_distance: float = 4.0*creature_radius
const creature_reproduce_time: float = 0.4
const creature_brain_variation: float32 = 0.05
const creature_speed: float = 10.0
const max_creatures: int = 256
const reincarnate_time: float = 3.0
const save_time: float = 30.0
const perceived_other_creatures = 8
const timestep: float = 1.0/15.0
const progress_filename = "progress.flatty_nim_library"

type 
  OtherCreatureInput = object
    relPos: Vec2
    hue: float32

  CreatureInput = object
    myPos: Vec2
    otherCreatures: array[perceived_other_creatures, OtherCreatureInput]
  CreatureInputData = array[2 + perceived_other_creatures*3, float32]

func toData(c: CreatureInput): CreatureInputData =
  result[0] = c.myPos.x
  result[1] = c.myPos.y
  for i, curC in c.otherCreatures.pairs:
    result[2+i*3] = curC.relPos.x
    result[2+i*3+1] = curC.relPos.y
    result[2+i*3+2] = curC.hue

# has to be before everything because of pixie name conflict

type
  CreatureBrain[T] = object
    gru: GRULayer[T]
    memory: Variable[Tensor[T]]
    fc1: Linear[T]
    fc2: Linear[T]
  Creature = object
    alive: bool
    col: Color
    brain: Option[CreatureBrain[float32]]
    babyTime: float
    aliveTime: float
    pos: Vec2

let ctx = arraymancer.newContext(Tensor[float32])

func byteLen[T](t: Tensor[T]): int =
  t.size * T.sizeof

proc toFlatty[T](s: var string, t: Variable[Tensor[T]]) =
  let isNil = t == nil
  s.toFlatty(isNil) 
  if isNil:
    return

  let bytelen: int = t.value.byteLen

  s.toFlatty(bytelen)
  s.toFlatty(t.value.shape)
  
  if bytelen > 0:
    s.setLen(s.len + bytelen)
    let dest = s[s.len - bytelen].addr
    copyMem(dest, t.value.get_data_ptr, bytelen)

proc fromFlatty[T: KnownSupportsCopyMem](s: string, i: var int, x: var Variable[Tensor[T]]) =
  var isNil: bool
  
  s.fromFlatty(i, isNil)
  if isNil:
    # x = ctx.variable(Tensor[T]())
    return

  var bytelen: int
  s.fromFlatty(i, bytelen)

  var newTensor: Tensor[T]
  var shape: Metadata
  s.fromFlatty(i, shape)
  # newTensor.fromBuffer(s[i].addr, shape, )
  var size: int
  newTensor.initTensorMetadata(size, shape, rowMajor)
  newTensor.storage.allocCPUStorage(size)
  
  assert(bytelen == newTensor.byteLen)
  
  # let byteLen = newTensor.size * T.sizeof
  # newTensor.copyFromRaw(s[i].unsafeAddr, byteLen)
  # i += byteLen

  if bytelen > 0:
    copyMem(cast[ptr UncheckedArray[char]](newTensor.unsafe_raw_buf()), s[i].unsafeAddr, bytelen)
    i += bytelen

  # let r_ptr = newTensor.unsafe_raw_buf()
  # for dataIndex in i..<(i+newTensor.size):
  #   r_ptr[dataIndex] = s[dataIndex]
  # i += newTensor.size

  x = ctx.variable(newTensor)

const gruHidden = 8
proc create[T](ctx: Context[Tensor[T]]): CreatureBrain[T] =
  const gruStack = 1
  result.gru = ctx.init(GRULayer[T], CreatureInputData.len, gruHidden, gruStack)
  result.memory = ctx.variable(zeros[T](gruStack, 1, gruHidden))
  result.fc1 = ctx.init(Linear[T], gruHidden, 8)
  result.fc2 = ctx.init(Linear[T], 8, 2)

proc think[T](creature: Creature, ctx: Context[T], input: CreatureInputData): Vec2 =
  # echo creature.brain.testLayer.forward( ctx.variable( randomTensor(1, CreatureInput.len, 1.0).astype(float32) )).value
  # echo creature.brain.testLayer.forward( ctx.variable([2.0.float32, 1.0.float32].toTensor().reshape(1,2) )).value
  let tensorInput = ctx.variable(input.toTensor().reshape(1, CreatureInputData.len))
  # print creature.brain.fc1.outShape, creature.brain.fc1.inShape, tensorInput.value.shape
  var brain = creature.brain.get().unsafeAddr
  
  # for name, value in brain.gru.fieldPairs:
  #   print name, value != nil
  #   print value.is_grad_needed
  let (gruOut, newMemory) = brain.gru.forward(tensorInput.reshape(1, 1, CreatureInputData.len), brain.memory)

  brain.memory = newMemory
  let memoryOutput = gruOut.reshape(1, gruHidden)

  let fc1 = (brain.fc1.forward(memoryOutput))
  let fc1activated = tanh(fc1)
  # let fc1 = (brain.fc1.forward(tensorInput))
  let fc2 = (brain.fc2.forward(fc1activated))
  let fc2activated = tanh(fc2)
  # print fc2activated.value.rank
  let output = fc2activated.value.reshape(2).toSeq1D()

  result = vec2(output[0], output[1])
  # echo output[0]

proc copyBrain[T](creature: var Creature, ctx: Context[T], source: Creature) =
  creature.brain = source.brain
  if source.brain.isSome():
    # template cloneLayer(theObject: untyped) =
    #   creature.theObject = ctx.variable(source.theObject.value.clone())
    # cloneLayer(brain.get().fc1.weight)

    var intoBrain = creature.brain.get().unsafeAddr
    var fromBrain = source.brain.get().unsafeAddr

    intoBrain.memory = ctx.variable(fromBrain.memory.value.clone())
    intoBrain.gru.w3s0 = ctx.variable(fromBrain.gru.w3s0.value.clone())
    intoBrain.gru.w3sN = ctx.variable(fromBrain.gru.w3sN.value.clone())
    intoBrain.gru.u3s = ctx.variable(fromBrain.gru.u3s.value.clone())
    intoBrain.gru.bW3s = ctx.variable(fromBrain.gru.bW3s.value.clone())
    intoBrain.gru.bU3s = ctx.variable(fromBrain.gru.bU3s.value.clone())

    intoBrain.fc1.weight = ctx.variable(fromBrain.fc1.weight.value.clone())
    intoBrain.fc1.bias = ctx.variable(fromBrain.fc1.bias.value.clone())
    intoBrain.fc2.weight = ctx.variable(fromBrain.fc2.weight.value.clone())
    intoBrain.fc2.bias = ctx.variable(fromBrain.fc2.bias.value.clone())

    # creature.brain.get().gru.weight = ctx.variable(source.brain.get().fc1.weight.value.clone())
    # creature.brain.get().gru.weight = ctx.variable(source.brain.get().fc1.weight.value.clone())
    # creature.brain.get().gru.weight = ctx.variable(source.brain.get().fc1.weight.value.clone())
  
    # creature.brain.get().fc1.weight = ctx.variable(source.brain.get().fc1.weight.value.clone())
    # creature.brain.get().fc1.bias = ctx.variable(source.brain.get().fc1.bias.value.clone())
    # creature.brain.get().fc2.weight = ctx.variable(source.brain.get().fc2.weight.value.clone())
    # creature.brain.get().fc2.bias = ctx.variable(source.brain.get().fc2.bias.value.clone())


proc mutateBrain(creature: var Creature) =
  proc mutateTensor(t: Variable[Tensor[float32]]) =
    let shape = t.value.shape.data[0..<t.value.shape.len]
    t.value += randomTensor[float32](shape, [-creature_brain_variation, creature_brain_variation])
  mutateTensor(creature.brain.get().gru.w3s0)  
  mutateTensor(creature.brain.get().gru.w3sN)  
  mutateTensor(creature.brain.get().gru.u3s)  
  mutateTensor(creature.brain.get().gru.bW3s)  
  mutateTensor(creature.brain.get().gru.bU3s)  

  mutateTensor(creature.brain.get().fc1.weight)
  mutateTensor(creature.brain.get().fc1.bias)
  mutateTensor(creature.brain.get().fc2.weight)
  mutateTensor(creature.brain.get().fc2.bias)

type Creatures = array[max_creatures, Creature]

import boxy, opengl, windy, random

let windowSize = ivec2(1280, 800)


let window = newWindow("RNN Evolution", windowSize)
makeContextCurrent(window)

loadExtensions()

let bxy = newBoxy()

let circleImg = newImage(32, 32)
block:
  let ctx = newContext(circleImg)
  ctx.fillStyle.color = color(0, 0, 0, 1)
  const outlineSize = 3.0
  let outerCircle = circle(vec2(circleImg.width.float, circleImg.height.float)/2.0, circleImg.width.float/2.0)
  ctx.fillCircle(outerCircle)
  ctx.fillStyle.color = color(1, 1, 1, 1)

  ctx.fillCircle(circle(outerCircle.pos, outerCircle.radius - outlineSize))

bxy.addImage("circle", circleImg)
proc draw(creature: Creature) =
  let desired_size = vec2(creature_radius*2.0)
  bxy.drawImage("circle", rect=rect(creature.pos-desired_size/2.0, desired_size), tint=creature.col)


func newCreature(creatures: var Creatures): Option[ptr Creature] =
  for c in creatures.mitems:
    if not c.alive:
      return some[ptr Creature](c.addr)
  return none[ptr Creature]()

proc babyCreature(c: Creature): Creature =
  let theta = rand(2.0*PI)
  result.alive = true
  result.pos =c.pos + vec2(cos(theta),sin(theta))*baby_distance
  result.col = c.col
  result.copyBrain(ctx, c)
  result.mutateBrain()
  proc randomizeColor(f: float): float = 
    result = f + rand(-0.1..0.1)
    result = clamp(result, 0.0, 1.0)
  result.col.r = randomizeColor(result.col.r)
  result.col.g = randomizeColor(result.col.g)
  result.col.b = randomizeColor(result.col.b)
  result.col.a = 1.0

var creatures: Creatures

proc initCreature(c: ptr Creature) =
  c.alive = true
  c.pos = vec2(rand(map_size), rand(map_size))
  c.col = creature_start_color
  c.brain = some[typeof(Creature.brain.val)](ctx.create())
  c.babyTime = 0.0

if fileExists(progress_filename):
  creatures = fromFlatty(readFile(progress_filename), Creatures)
else:
  # initialize some creatures
  assert(init_creatures < max_creatures)
  randomize()
  for _ in 0..init_creatures-1:
    let c = creatures.newCreature().get()
    c.initCreature()

var reincarnationTimer = 0.0

proc processCreatures(creatures: var Creatures, delta: float) =
  reincarnationTimer += delta
  var toReincarnate = -1
  if reincarnationTimer > reincarnate_time:
    toReincarnate = rand(creatures.len-1)
    reincarnationTimer = 0.0
  for i, c in creatures.mpairs:
    if i == toReincarnate:
      c.addr.initCreature()
    if c.alive:
      c.babyTime += delta
      if c.babyTime > creature_reproduce_time:
        c.babyTime = 0.0
        let possibleCreature = creatures.newCreature()
        if possibleCreature.isSome():
          possibleCreature.get()[] = babyCreature(c)
      
      profile "See the world":
        var toSort: seq[OtherCreatureInput]

        for ii, otherC in creatures.pairs:
          if ii == i or not c.alive:
            continue
          # from zero to one, how far away and in what direction
          let rel = otherC.pos - c.pos
          if rel.lengthSq < creature_radius*creature_radius:
            c.alive = false
            creatures[ii].alive = false
            break

          let relPos = rel/map_size
          
          toSort.add(OtherCreatureInput(
            relPos: relPos,
            hue: otherC.col.asHsv().h,
          ))
        if not c.alive:
          continue
        toSort.sort(proc (x, y: OtherCreatureInput): int =
          system.cmp[float32](x.relPos.lengthSq, y.relPos.lengthSq)
        )

        var inputData = CreatureInput(myPos: c.pos/map_size)
        for i in 0..high(inputData.otherCreatures):
          if i < toSort.len:
            inputData.otherCreatures[i] = toSort[i]
          else:
            inputData.otherCreatures[i] = OtherCreatureInput(
              relPos: vec2(1.0, 1.0),
              hue: 1.0,
            )
      # c.pos = c.pos + vec2(rand(2.0)-1.0, rand(2.0)-1.0).normalized()*creature_speed*delta
      var movement = vec2(0.0)
      profile "Creature thinking":
        ctx.no_grad_mode:
            movement = c.think(ctx, inputData.toData())
      # movement = vec2(1.0, 0.0)
      if movement.length < 0.0001:
        movement = vec2(0.0)
      else:
        movement = movement.normalize()
      c.pos = c.pos + movement*creature_speed*delta
      proc wrapPos(pos: float): float =
        if pos < 0.0:
          result = map_size + pos
        elif pos > map_size:
          result = pos - map_size
        else:
          result = pos
      c.pos.x = wrapPos(c.pos.x)
      c.pos.y = wrapPos(c.pos.y)
      # if c.pos.x < 0.0 or c.pos.x > map_size or
      #    c.pos.y < 0.0 or c.pos.y > map_size:
      #   c.alive = false
      # else:
      c.aliveTime += delta

var render: bool = true

window.onButtonPress = proc(button: Button) =
  if button == KeyZ:
    render = not render

var beganTime = cpuTime()
var printedTime = cpuTime()
var processedTime = 0.0

window.onFrame = proc() =
  let start = cpuTime()
  while cpuTime() - start <= 1.0/60.0:
    processCreatures(creatures, timestep)
    processedTime += timestep

  if cpuTime() - printedTime > save_time:
    # printresults()
    var avgAge: float = 0.0
    for c in creatures:
      avgAge += c.aliveTime
    avgAge /= creatures.len().float
    echo processedTime, " ", avgAge
    printedTime = cpuTime()
    writeFile(progress_filename, toFlatty(creatures))


  # Clear the screen and begin a new frame.
  bxy.beginFrame(window.size)


  bxy.saveTransform()
  bxy.scale(vec2(3,3))
  bxy.drawRect(rect(0,0,map_size, map_size), map_color)
  for c in creatures:
    if c.alive:
      c.draw()
  bxy.restoreTransform()

  bxy.endFrame()
  window.swapBuffers()


while not window.closeRequested:
  pollEvents()