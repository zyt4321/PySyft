"""The purpose of this application is to allow us to dev and test PySyft
functionality on an actual local network. This is NOT meant to be run in
production (that's the *actual* grid's job)."""


# stdlib
import binascii
import json
import pickle

# third party
import flask
from flask import Flask
from flask import Response
from nacl.encoding import HexEncoder

# syft absolute
from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.common.serde.deserialize import _deserialize
from syft.core.node.domain.domain import Domain
from syft.grid.services.signaling_service import PullSignalingService
from syft.grid.services.signaling_service import PushSignalingService
from syft.grid.services.signaling_service import RegisterDuetPeerService
import asyncio

try:
    # stdlib
    from asyncio import get_running_loop  # noqa Python >=3.7
except ImportError:  # pragma: no cover
    # stdlib
    from asyncio.events import _get_running_loop as get_running_loop  # pragma: no cover


loop = asyncio.new_event_loop()

app = Flask(__name__)

from loguru import logger

LOG_FILE = "syft_do.log"
logger.add(
    LOG_FILE, enqueue=True, colorize=False, diagnose=True, backtrace=True, level="TRACE"
)


domain = Domain(name="ucsf")
duet = domain.get_root_client()
accept_handler = {
    "timeout_secs": -1,
    "action": "accept",
    "print_local": True,
    "log_local": True,
}
duet.requests.add_handler(accept_handler)


@app.route("/metadata")
def get_metadata() -> flask.Response:
    metadata = domain.get_metadata_for_client()
    metadata_proto = metadata.serialize()
    r = Response(
        response=metadata_proto.SerializeToString(),
        status=200,
    )
    r.headers["Content-Type"] = "application/octet-stream"
    return r


@app.route("/", methods=["POST"])
def process_domain_msgs() -> flask.Response:
    data = flask.request.get_data()
    obj_msg = _deserialize(blob=data, from_bytes=True)
    if isinstance(obj_msg, SignedImmediateSyftMessageWithReply):
        print(
            f"Signaling server SignedImmediateSyftMessageWithReply: {obj_msg.message} watch"
        )
        reply = domain.recv_immediate_msg_with_reply(msg=obj_msg)
        r = Response(response=reply.serialize(to_bytes=True), status=200)
        r.headers["Content-Type"] = "application/octet-stream"
        return r
    elif isinstance(obj_msg, SignedImmediateSyftMessageWithoutReply):
        print(
            f"Signaling server SignedImmediateSyftMessageWithoutReply: {obj_msg.message} watch"
        )
        domain.recv_immediate_msg_without_reply(msg=obj_msg)
        r = Response(status=200)
        return r
    else:
        print(
            f"Signaling server SignedImmediateSyftMessageWithoutReply: {obj_msg.message} watch"
        )
        domain.recv_eventual_msg_without_reply(msg=obj_msg)
        r = Response(status=200)
        return r


# async def async_run():
#     print("async run")
#     global domain
#     # asyncio.ensure_future(domain.run_handlers())
#     await asyncio.sleep(5)
#     print(asyncio.all_tasks())
#     app.run()
#     print("async run app rnuning")


def run() -> None:
    app.run(threaded=True)
