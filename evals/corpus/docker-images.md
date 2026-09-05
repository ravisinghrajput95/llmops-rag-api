# Container image construction

A multi-stage build compiles dependencies in one stage and copies only the
resulting artefacts into a smaller runtime stage. Build tools, compilers and
caches stay out of the final image.

Image architecture must match the host. Cloud Run executes x86_64 images only,
so a build on an ARM machine must specify the target platform explicitly or the
container fails at startup with an exec format error.

Layer ordering determines cache reuse. Copying a dependency manifest and
installing dependencies before copying application source means a source change
does not invalidate the dependency layer, which is usually the slowest to
rebuild.

Running as a non-root user is the default expectation for a production image. A
container that must write at runtime should write to a path owned by that user,
not to the application directory.
