# Package

version       = "0.1.0"
author        = "Cameron Reikes"
description   = "A new awesome nimble package"
license       = "MIT"
srcDir        = "src"
bin           = @["recurrent_evolution"]


# Dependencies

requires "nim >= 1.6.6"
requires "windy"
requires "boxy"
requires "pixie"
requires "arraymancer == 0.7.15"
requires "print"
requires "chroma"
requires "vmath"
requires "flatty"
requires "kdtree"
requires "https://github.com/jblindsay/kdtree"