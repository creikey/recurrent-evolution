import arraymancer, print, chroma, vmath, chroma/transformations, algorithm, flatty, os, options, kdtree, std/monotimes

import profile

proc now(): float64 =
  ## Gets current time
  getMonoTime().ticks.float64 / 1000000000.0

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
const reincarnate_time: float = 5.0
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
  World = object
    creatures: array[max_creatures, Creature]
    totalTime: float

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
const gruStack = 1
proc create[T](ctx: Context[Tensor[T]]): CreatureBrain[T] =
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

proc draw(creature: Creature, beingHovered: bool) =
  var desiredSize = vec2(creature_radius*2.0)
  if beingHovered: desiredSize *= 1.5
  bxy.drawImage("circle", rect=rect(creature.pos-desiredSize/2.0, desiredSize), tint=creature.col)


func newCreature(world: var World): Option[ptr Creature] =
  for c in world.creatures.mitems:
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


var world: World

proc initCreature(c: ptr Creature) =
  c.alive = true
  c.pos = vec2(rand(map_size), rand(map_size))
  c.col = creature_start_color
  c.brain = some[typeof(Creature.brain.val)](ctx.create())
  c.babyTime = 0.0

if fileExists(progress_filename):
  world = fromFlatty(readFile(progress_filename), World)
else:
  # initialize some creatures
  assert(init_creatures < max_creatures)
  randomize()
  for _ in 0..init_creatures-1:
    let c = world.newCreature().get()
    c.initCreature()

var reincarnationTimer = 0.0

proc processWorld(world: var World, delta: float) =
  world.totalTime += delta
  reincarnationTimer += delta
  var toReincarnate = -1
  if reincarnationTimer > reincarnate_time:
    toReincarnate = rand(world.creatures.len-1)
    reincarnationTimer = 0.0

  profile "KdTree construction":
    var creaturePoints = newSeq[KdPoint]()
    var creatureValues = newSeq[Creature]()
    for c in world.creatures:
      if c.alive:
        creaturePoints.add([c.pos.x.float, c.pos.y.float].KdPoint)
        creatureValues.add(c)
    var tree = newKdTree[Creature](creaturePoints, creatureValues)

  for i, c in world.creatures.mpairs:
    if i == toReincarnate:
      c.addr.initCreature()
    if c.alive:
      c.babyTime += delta
      if c.babyTime > creature_reproduce_time:
        c.babyTime = 0.0
        let possibleCreature = world.newCreature()
        if possibleCreature.isSome():
          possibleCreature.get()[] = babyCreature(c)
      
      when false: profile "See the world":
        var closestCreatures: seq[OtherCreatureInput]

        for ii, otherC in world.creatures.pairs:
          if ii == i or not c.alive:
            continue
          # from zero to one, how far away and in what direction
          let rel = otherC.pos - c.pos
          if rel.lengthSq < creature_radius*creature_radius:
            c.alive = false
            world.creatures[ii].alive = false
            break

          let relPos = rel/map_size
          
          closestCreatures.add(OtherCreatureInput(
            relPos: relPos,
            hue: otherC.col.asHsv().h,
          ))
        if not c.alive:
          continue
        closestCreatures.sort(proc (x, y: OtherCreatureInput): int =
          system.cmp[float32](x.relPos.lengthSq, y.relPos.lengthSq)
        )

      var inputData = CreatureInput(myPos: c.pos/map_size)

      var closestCreatures = tree.nearestNeighbours([c.pos.x.float, c.pos.y.float], inputData.otherCreatures.len)

      for i in 0..high(inputData.otherCreatures):
        if i < closestCreatures.len:
          let closestResult: Creature = closestCreatures[i][1]
          inputData.otherCreatures[i] = OtherCreatureInput(
            relPos: (closestResult.pos - c.pos)/map_size,
            hue: closestResult.col.asHsv().h,
          )
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
      profile "Wrapping":
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
        # c.pos.x = wrapPos(c.pos.x)
        # c.pos.y = wrapPos(c.pos.y)
        if c.pos.x < 0.0 or c.pos.x > map_size or
          c.pos.y < 0.0 or c.pos.y > map_size:
          c.alive = false
        else:
          c.aliveTime += delta

var asFastAsPossible: bool = false
var lastTickTime = now()

type Camera = object
  translation: Vec2
  scale: float

func screenToWorld(c: Camera): GMat3[float32] =
  scale(vec2(c.scale)) * translate(c.translation)

func worldToScreen(c: Camera): GMat3[float32] =
  c.screenToWorld().inverse()

var cam = Camera(scale: 1.0 / 3.0)
var printedTime = now()
var processedTime = 0.0

window.onButtonPress = proc(button: Button) =
  if button == KeySpace:
    asFastAsPossible = not asFastAsPossible
    if not asFastAsPossible:
      lastTickTime = now()
  elif button == KeyP:
    printresults()

window.onFrame = proc() =
  if window.buttonDown[MouseMiddle] or window.buttonDown[MouseRight]:
    cam.translation -= window.mouseDelta.vec2

  block:
    let zoomDelta = 1.0 + (-window.scrollDelta.y)*0.1
    let before = cam.screenToWorld() * window.mousePos.vec2
    cam.scale *= zoomDelta
    cam.translation += (cam.worldToScreen() * before) - window.mousePos.vec2

  var hoveringIndex = -1
  for i in countdown(high(world.creatures), low(world.creatures)): # reverse order so creature drawn on top gets mouse hover
    let c = world.creatures[i]
    if (cam.screenToWorld() * window.mousePos.vec2).dist(c.pos) < creature_radius:
      hoveringIndex = i
      break

  if asFastAsPossible:
    let start = now()
    while now() - start <= 1.0/60.0:
      processWorld(world, timestep)
      processedTime += timestep
  else:
    let curNow = now()
    while curNow > lastTickTime:
      processWorld(world, timestep)
      lastTickTime += timestep
      processedTime += timestep
  
  if now() - printedTime > save_time:
    var avgAge: float = 0.0
    var bestCreatureMemory: Variable[Tensor[float32]] = nil
    var longestLife: float = 0.0
    for c in world.creatures:
      if c.alive:
        avgAge += c.aliveTime
        if c.aliveTime > longestLife:
          longestLife = c.aliveTime
          bestCreatureMemory = c.brain.get().memory
    avgAge /= world.creatures.len().float

    assert(gruStack == 1)
    echo processedTime, "|", avgAge, "|", longestLife, "|", bestCreatureMemory.value.reshape(gruHidden).toSeq1D()
    printedTime = now()
    writeFile(progress_filename, toFlatty(world))


  # Clear the screen and begin a new frame.
  bxy.beginFrame(window.size)

  bxy.saveTransform()
  bxy.applyTransform(cam.worldToScreen()) # draw in world coordinates
  bxy.drawRect(rect(0,0,map_size, map_size), map_color)
  for i, c in world.creatures.pairs:
    if c.alive:
      c.draw(i == hoveringIndex)
  bxy.restoreTransform()

  bxy.endFrame()
  window.swapBuffers()


while not window.closeRequested:
  pollEvents()