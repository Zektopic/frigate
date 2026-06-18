BOARDS += xdnpu

local-xdnpu: version
	docker buildx bake --file=docker/xdnpu/xdnpu.hcl xdnpu \
		--set xdnpu.tags=frigate:latest-xdnpu \
		--load

build-xdnpu: version
	docker buildx bake --file=docker/xdnpu/xdnpu.hcl xdnpu \
		--set xdnpu.tags=$(IMAGE_REPO):${GITHUB_REF_NAME}-$(COMMIT_HASH)-xdnpu \
		--set xdnpu.tags=$(IMAGE_REPO):latest-xdnpu

push-xdnpu: build-xdnpu
	docker buildx bake --file=docker/xdnpu/xdnpu.hcl xdnpu \
		--set xdnpu.tags=$(IMAGE_REPO):${GITHUB_REF_NAME}-$(COMMIT_HASH)-xdnpu \
		--set xdnpu.tags=$(IMAGE_REPO):latest-xdnpu \
		--push
