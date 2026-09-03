# Database for InteFL Framework

___

## Storage modes

`EXPERIMENT_DB_MODE` selects where experiment results go:

| Mode | Destination | Credentials required |
|---|---|---|
| `local` | SQLite file on this machine (default: `out/experiments.db`) | none |
| `server` | The research group's experiment server, over HTTP | `EXPERIMENT_SERVER_URL`, `EXPERIMENT_SERVER_TOKEN` |
| `direct` | A PostgreSQL database, by URL | `EXPERIMENT_DB_*` |
| `neon` | Neon cloud DBMS | `EXPERIMENT_DB_*` |

`local` is the default when nothing is configured, and `server` mode **falls back to
`local`** when no server URL and token are available or when the server rejects the
token. A simulation therefore always persists its results, with or without access to
the group's infrastructure. Either way, CSVs and plots are written to `out/` independently
of the database.

The local database is created from `models.py` on first use and stamped at the current
Alembic revisions, so `alembic upgrade heads` stays usable afterwards. Delete the file to
start over.

___

## .env variables

* EXPERIMENT_DB_MODE: `local`, `server`, `direct` or `neon` (see above)
* EXPERIMENT_DB_LOCAL_PATH: path of the SQLite file for `local` mode, default `out/experiments.db`
* EXPERIMENT_DB_NAME
* EXPERIMENT_DB_PASSWORD
* EXPERIMENT_DB_HOST
* EXPERIMENT_DB_PORT
* EXPERIMENT_DB_USER
* FULL_AUTHOR_NAME: name of the experiment author, defaults to `git config user.name`
* AUTHOR_EMAIL: defaults to `git config user.email`
* EXPERIMENT_SERVER_URL: host where experiment server is located
* EXPERIMENT_SERVER_TOKEN: access token provided by admin
* EXPERIMENT_SERVER_PORT

Missing values are prompted for on an interactive terminal and saved to `.env.private`.
Headless runs never prompt: they use the documented defaults instead, so a simulation
started as a subprocess cannot hang waiting on stdin.

___

## Joining the research group

A new member cloning the private repository starts with `.env.public` (mode `server` and
the server URL) but no token, so nothing uploads yet. To connect:

```sh
echo "EXPERIMENT_SERVER_TOKEN=<token issued by an admin>" >> .env.private
```

Or simply run `sh run_simulation.sh`, or start the frontend API server, from a terminal:
`initialize_remote_access()` prompts for the token once and saves it to `.env.private`.
Tokens are issued with `python -m src.utils.generate_db_token` by someone with direct
database access.

The frontend API server resolves the token at startup and publishes it to the process
environment, so simulations launched from the UI inherit it and upload to the same
server. That resolution has to happen at startup rather than per request, because a
request handler must never block the event loop waiting on stdin.

If the token is missing or rejected, the run logs a warning naming the reason and stores
its results locally instead of failing.
