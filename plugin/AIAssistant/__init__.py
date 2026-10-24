# AI Assistant plugin for STE Slicer.
#
# Only `register` imports the slicer-dependent part, so the subpackages
# (printplan, agent) can be imported and tested outside the slicer.


def getMetaData():
    return {}


def register(app):
    from . import AIAssistant
    return {"extension": AIAssistant.AIAssistant()}
