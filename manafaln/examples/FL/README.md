# NVFlare FL Example

## Usage

### Before Start

Use Provided [Dockerfile](https://gitlab.com/nanaha1003/manafaln/-/blob/master/Dockerfile.nvflare) to setup environment.
Or manually install all required dependencies.

The provisioning scripts contains in `provision` directory.
Two directories `server` and `client` contains configurations and code for FL client and server.

### Provision

Go to `provision` and modify `project.yaml` according to your setting. Then use `provision.sh` to generate startup kits for 
the server and each client. The startup kits will placed in `build/[project_name]/prod_[xx]`.

The startup kits should be placed on each server. In this example we simulate multiple servers via docker virtualization.
A startup script `start_fl.sh` is provided to create a tmux session for managing all the clients.

To use the `start_fl.sh` you will need to manually put the startup kits into `sites` directory and modify the line 6-8 `ADMIN`, `SERVER`, `CLIENTS` to
according directory name under `sites`. Then you have to make sure the `docker.sh` in each startup kit is correct (Check for image name, tag and volume mount, network options).

If `start_fl.sh` is successfully executed, a tmux session will be created, each client will connect to the server and the server will show the number of clients registered.

### Code

Put the code (`server` and `client` directories) in `transfer` directory under the admin client directory. You will be able to deploy the code to the `server` and `client`
via admin interface.

The training/validation code is in `client/custom`, the configuration is `config`. In `config`, `config_fed_client.json` and `config_fed_server.json` are for NVFlare,
`config_train.json` and `config_validation.json` are for `manafaln`. Please make sure the data settings (path) match the real directory layout.

### Running the example

Goto the `ADMIN` directory and run `startup/fl_admin.sh`, login with the user ID (email) defined in `project.yml`.
Use `?` will show all available commands.

