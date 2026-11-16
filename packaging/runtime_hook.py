"""Runs before application imports, including Hugging Face's cached constants."""

from bundle_runtime import configure_offline

configure_offline()
