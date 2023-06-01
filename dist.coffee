#!/usr/bin/env coffee

> @w5/mdt/make.js
  @w5/uridir

ROOT = uridir(import.meta)
await make ROOT
process.exit()
