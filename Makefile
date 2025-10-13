# Do not remove this block. It is used by the 'help' rule when
# constructing the help output.
# help:
# help: IntronDB Makefile help
# help:

.PHONY: help
# help: help				- Please use "make <target>" where <target> is one of
help:
	@grep "^# help\:" Makefile | sed 's/\# help\: //' | sed 's/\# help\://'

.PHONY: e
# help: e				- cp env.example .env
e:
	@cp env.example .env


UNAME := $(shell uname)
