#  Copyright (c) 2022 by Amplo.

import requests

from amplo.utils import check_dtypes

__all__ = ["BaseRequestAPI"]


class BaseRequestAPI:
    """
    Base class for API.

    Parameters
    ----------
    host : str
        API host.
    access_token : str
        API access token.
    """

    def __init__(self, host: str, access_token: str):
        check_dtypes(("host", host, str), ("access_token", access_token, str))
        self.host = host
        self.access_token = access_token

    def __repr__(self):
        """
        Readable string representation of the class.
        """

        return f"{self.__class__.__name__}({self.host.removeprefix('https://')})"

    def _authorization_header(self) -> dict:
        raise NotImplementedError("Abstract method.")

    def request(self, method: str, action: str, **kwargs) -> requests.Response:
        """
        Send a request to the API.

        Parameters
        ----------
        method : str
            Request method (``GET``, ``PUT``, ``POST``, ...).
        action : str
            Path to API action.
        kwargs : Any, optional
            Additional keyword arguments for the request (``params``, ``headers``, ...).

        Returns
        -------
        requests.Response
            The request's response.

        Raises
        ------
        requests.HTTPError
            When the request's response has another status code than 200.
        """

        # Add authorization string to "headers" key
        headers: dict = kwargs.get("headers") or {}
        check_dtypes("kwargs__headers", headers, dict)
        headers.update(self._authorization_header())

        # Request
        url = f"{self.host}/api/{action.lstrip('/')}"
        response = requests.request(method, url, headers=headers, **kwargs)

        # Verify response
        if not response.status_code == 200:
            raise requests.HTTPError(f"{response} {response.text}")

        return response
