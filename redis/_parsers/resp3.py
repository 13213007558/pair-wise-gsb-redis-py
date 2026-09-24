from logging import getLogger
from typing import Any, Union

from ..exceptions import ConnectionError, InvalidResponse, ResponseError
from ..typing import EncodableT
from ..utils import SENTINEL
from .base import (
    AsyncPushNotificationsParser,
    AttributedResponse,
    PushNotificationsParser,
    _AsyncRESPBase,
    _RESPBase,
)
from .socket import SERVER_CLOSED_CONNECTION_ERROR


class _RESP3Parser(_RESPBase, PushNotificationsParser):
    """RESP3 protocol implementation"""

    def __init__(self, socket_read_size):
        super().__init__(socket_read_size)
        self.pubsub_push_handler_func = self.handle_pubsub_push_response
        self.node_moving_push_handler_func = None
        self.maintenance_push_handler_func = None
        self.oss_cluster_maint_push_handler_func = None
        self.invalidation_push_handler_func = None

    def handle_pubsub_push_response(self, response):
        logger = getLogger("push_response")
        logger.debug("Push response: %s", response)
        return response

    def read_response(
        self,
        disable_decoding=False,
        push_request=False,
        timeout: Union[float, object] = SENTINEL,
        return_attributes=False,
    ):
        pos = self._buffer.get_pos() if self._buffer is not None else None
        try:
            result = self._read_response(
                disable_decoding=disable_decoding,
                push_request=push_request,
                timeout=timeout,
                return_attributes=return_attributes,
            )
        except BaseException:
            if self._buffer is not None:
                self._buffer.rewind(pos)
            raise
        else:
            if self._buffer is not None:
                try:
                    self._buffer.purge()
                except AttributeError:
                    # Buffer may have been set to None by another thread after
                    # the check above; result is still valid so we don't raise
                    pass
            if return_attributes and not isinstance(result, AttributedResponse):
                # the reply carried no attribute frame; report empty metadata
                result = AttributedResponse(result, None)
            return result

    def _read_response(
        self,
        disable_decoding=False,
        push_request=False,
        timeout: Union[float, object] = SENTINEL,
        return_attributes=False,
    ):
        attributes = None
        while True:
            raw = self._buffer.readline(timeout=timeout)
            if not raw:
                raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR)

            byte, response = raw[:1], raw[1:]

            # attribute frame: metadata describing the reply that follows it
            if byte == b"|":
                frame_attributes = {}
                for _ in range(int(response)):
                    key = self._read_response(
                        disable_decoding=disable_decoding, timeout=timeout
                    )
                    frame_attributes[key] = self._read_response(
                        disable_decoding=disable_decoding, timeout=timeout
                    )
                if attributes is None:
                    attributes = frame_attributes
                else:
                    # consecutive attribute frames describe the same reply
                    attributes.update(frame_attributes)
                continue

            # push response
            if byte == b">":
                response = [
                    self._read_response(
                        disable_decoding=disable_decoding,
                        push_request=push_request,
                        timeout=timeout,
                        return_attributes=return_attributes,
                    )
                    for _ in range(int(response))
                ]
                response = self.handle_push_response(response)

                # if this is a push request return the push response
                if push_request:
                    if attributes is not None and return_attributes:
                        return AttributedResponse(response, attributes)
                    return response

                # an out-of-band push is handed to the push handler and does
                # not consume the reply slot of the command being read, and
                # attributes collected so far still belong to the reply that
                # follows the push. The loop also keeps forwarding `timeout`:
                # it is the caller's bound for this read, and dropping it
                # would silently fall back to the connection's socket_timeout
                # for the rest of the response just because a push message
                # happened to interleave.
                continue

            break

        # server returned an error
        if byte in (b"-", b"!"):
            if byte == b"!":
                response = self._buffer.read(int(response), timeout=timeout)
            response = response.decode("utf-8", errors="replace")
            error = self.parse_error(response)
            # if the error is a ConnectionError, raise immediately so the user
            # is notified
            if isinstance(error, ConnectionError):
                raise error
            # otherwise, we're dealing with a ResponseError that might belong
            # inside a pipeline response. the connection's read_response()
            # and/or the pipeline's execute() will raise this error if
            # necessary, so just return the exception instance here.
            response = error
        # single value
        elif byte == b"+":
            pass
        # null value
        elif byte == b"_":
            response = None
        # int and big int values
        elif byte in (b":", b"("):
            response = int(response)
        # double value
        elif byte == b",":
            response = float(response)
        # bool value
        elif byte == b"#":
            response = response == b"t"
        # bulk response
        elif byte == b"$":
            response = self._buffer.read(int(response), timeout=timeout)
        # verbatim string response
        elif byte == b"=":
            response = self._buffer.read(int(response), timeout=timeout)[4:]
        # array response
        elif byte == b"*":
            response = [
                self._read_response(
                    disable_decoding=disable_decoding,
                    timeout=timeout,
                    return_attributes=return_attributes,
                )
                for _ in range(int(response))
            ]
        # set response
        elif byte == b"~":
            # redis can return unhashable types (like dict) in a set,
            # so we return sets as list, all the time, for predictability
            response = [
                self._read_response(
                    disable_decoding=disable_decoding,
                    timeout=timeout,
                    return_attributes=return_attributes,
                )
                for _ in range(int(response))
            ]
        # map response
        elif byte == b"%":
            # We cannot use a dict-comprehension to parse stream.
            # Evaluation order of key:val expression in dict comprehension only
            # became defined to be left-right in version 3.8
            resp_dict = {}
            for _ in range(int(response)):
                key = self._read_response(
                    disable_decoding=disable_decoding,
                    timeout=timeout,
                    return_attributes=return_attributes,
                )
                resp_dict[key] = self._read_response(
                    disable_decoding=disable_decoding,
                    push_request=push_request,
                    timeout=timeout,
                    return_attributes=return_attributes,
                )
            response = resp_dict
        else:
            raise InvalidResponse(f"Protocol Error: {raw!r}")

        if isinstance(response, bytes) and disable_decoding is False:
            response = self.encoder.decode(response)

        if attributes is not None and return_attributes:
            return AttributedResponse(response, attributes)
        return response


class _AsyncRESP3Parser(_AsyncRESPBase, AsyncPushNotificationsParser):
    def __init__(self, socket_read_size):
        super().__init__(socket_read_size)
        self.pubsub_push_handler_func = self.handle_pubsub_push_response
        self.invalidation_push_handler_func = None

    async def handle_pubsub_push_response(self, response):
        logger = getLogger("push_response")
        logger.debug("Push response: %s", response)
        return response

    async def read_response(
        self,
        disable_decoding: bool = False,
        push_request: bool = False,
        return_attributes: bool = False,
    ):
        if not self._connected:
            raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR)
        if self._chunks:
            # augment parsing buffer with previously read data
            self._buffer += b"".join(self._chunks)
            self._chunks.clear()
        self._pos = 0
        response = await self._read_response(
            disable_decoding=disable_decoding,
            push_request=push_request,
            return_attributes=return_attributes,
        )
        # Successfully parsing a response allows us to clear our parsing buffer
        self._clear()
        if return_attributes and not isinstance(response, AttributedResponse):
            # the reply carried no attribute frame; report empty metadata
            response = AttributedResponse(response, None)
        return response

    async def _read_response(
        self,
        disable_decoding: bool = False,
        push_request: bool = False,
        return_attributes: bool = False,
    ) -> Union[EncodableT, ResponseError, AttributedResponse, None]:
        if not self._stream or not self.encoder:
            raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR)
        response: Any
        attributes = None
        while True:
            raw = await self._readline()
            byte, response = raw[:1], raw[1:]

            # attribute frame: metadata describing the reply that follows it
            if byte == b"|":
                frame_attributes = {}
                for _ in range(int(response)):
                    key = await self._read_response(disable_decoding=disable_decoding)
                    frame_attributes[key] = await self._read_response(
                        disable_decoding=disable_decoding
                    )
                if attributes is None:
                    attributes = frame_attributes
                else:
                    # consecutive attribute frames describe the same reply
                    attributes.update(frame_attributes)
                continue

            # push response
            if byte == b">":
                response = [
                    (
                        await self._read_response(
                            disable_decoding=disable_decoding,
                            push_request=push_request,
                            return_attributes=return_attributes,
                        )
                    )
                    for _ in range(int(response))
                ]
                response = await self.handle_push_response(response)

                # if this is a push request return the push response
                if push_request:
                    if attributes is not None and return_attributes:
                        return AttributedResponse(response, attributes)
                    return response

                # an out-of-band push is handed to the push handler and does
                # not consume the reply slot of the command being read, and
                # attributes collected so far still belong to the reply that
                # follows the push.
                continue

            break

        # server returned an error
        if byte in (b"-", b"!"):
            if byte == b"!":
                response = await self._read(int(response))
            response = response.decode("utf-8", errors="replace")
            error = self.parse_error(response)
            # if the error is a ConnectionError, raise immediately so the user
            # is notified
            if isinstance(error, ConnectionError):
                self._clear()  # Successful parse
                raise error
            # otherwise, we're dealing with a ResponseError that might belong
            # inside a pipeline response. the connection's read_response()
            # and/or the pipeline's execute() will raise this error if
            # necessary, so just return the exception instance here.
            response = error
        # single value
        elif byte == b"+":
            pass
        # null value
        elif byte == b"_":
            response = None
        # int and big int values
        elif byte in (b":", b"("):
            response = int(response)
        # double value
        elif byte == b",":
            response = float(response)
        # bool value
        elif byte == b"#":
            response = response == b"t"
        # bulk response
        elif byte == b"$":
            response = await self._read(int(response))
        # verbatim string response
        elif byte == b"=":
            response = (await self._read(int(response)))[4:]
        # array response
        elif byte == b"*":
            response = [
                (
                    await self._read_response(
                        disable_decoding=disable_decoding,
                        return_attributes=return_attributes,
                    )
                )
                for _ in range(int(response))
            ]
        # set response
        elif byte == b"~":
            # redis can return unhashable types (like dict) in a set,
            # so we always convert to a list, to have predictable return types
            response = [
                (
                    await self._read_response(
                        disable_decoding=disable_decoding,
                        return_attributes=return_attributes,
                    )
                )
                for _ in range(int(response))
            ]
        # map response
        elif byte == b"%":
            # We cannot use a dict-comprehension to parse stream.
            # Evaluation order of key:val expression in dict comprehension only
            # became defined to be left-right in version 3.8
            resp_dict = {}
            for _ in range(int(response)):
                key = await self._read_response(
                    disable_decoding=disable_decoding,
                    return_attributes=return_attributes,
                )
                resp_dict[key] = await self._read_response(
                    disable_decoding=disable_decoding,
                    push_request=push_request,
                    return_attributes=return_attributes,
                )
            response = resp_dict
        else:
            raise InvalidResponse(f"Protocol Error: {raw!r}")

        if isinstance(response, bytes) and disable_decoding is False:
            response = self.encoder.decode(response)

        if attributes is not None and return_attributes:
            return AttributedResponse(response, attributes)
        return response
