"""The one number that says which QRUDO this is.

The repository has commits, but a customer's app has no repository --
it has whatever was frozen into it on build day.  This is that fact,
in a form a release page can be compared against: bump it when cutting
a release, tag the commit to match (v0.1.0), and updates.py does the
comparing on every customer's machine.
"""

VERSION = "0.1.0"
