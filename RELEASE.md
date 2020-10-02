# Release 0.3.0

The first release of the new backwards-incompatible API.

## Major Changes

* The main difference is the new `protoflow.layers.prototypes` which makes the
  distance layers obsolete. This makes implementing Siamese architectures much
  easier. Distances are however still available as functions.

* The second big change is that `protoflow.applications` are now written by
  subclassing `tf.keras.Model`. This makes the code a lot cleaner. However,
  model serialization is no longer possible. The work around is to only save the
  weights with `model.save_weights()`. Instead of deserializing the entire
  model, simply create a new model and load the saved the weights.

* In order to highlight the breaking changes in the documentation, there is a
  new logo.

## New Features

* New benchmark and toy datasets are now available under `protoflow.datasets`.

# Release 0.2.x

Bug fixes to the legacy API. No new features will be added to 0.2.x.

# Release 0.2.0

This release is meant to prepare for the upcoming breaking changes in the next
0.3.0 release, which will essentially be a backwards-incompatible fork.

# Release 0.1.0

## Bug Fixes

* Lots of fixes to setup.py including aesthetic fixes.
* Fix version number in the package init file.

# Release 0.0.1

Initial public release of ProtoFlow.
