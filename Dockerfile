FROM ubuntu:18.04

MAINTAINER Jacky Cao <jacky.cao@oulu.fi>

COPY . .

RUN ./gpu_fv l 5 51717

