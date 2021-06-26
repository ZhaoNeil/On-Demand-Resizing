/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"context"
	"flag"
	"os"
	"time"

	apiv1 "k8s.io/api/core/v1"
	"k8s.io/autoscaler/vertical-pod-autoscaler/common"
	vpa_clientset "k8s.io/autoscaler/vertical-pod-autoscaler/pkg/client/clientset/versioned"
	"k8s.io/autoscaler/vertical-pod-autoscaler/pkg/target"
	updater "k8s.io/autoscaler/vertical-pod-autoscaler/pkg/updater/logic"
	"k8s.io/autoscaler/vertical-pod-autoscaler/pkg/updater/priority"
	"k8s.io/autoscaler/vertical-pod-autoscaler/pkg/utils/limitrange"
	"k8s.io/autoscaler/vertical-pod-autoscaler/pkg/utils/metrics"
	metrics_updater "k8s.io/autoscaler/vertical-pod-autoscaler/pkg/utils/metrics/updater"
	"k8s.io/autoscaler/vertical-pod-autoscaler/pkg/utils/status"
	vpa_api_util "k8s.io/autoscaler/vertical-pod-autoscaler/pkg/utils/vpa"
	"k8s.io/client-go/informers"
	kube_client "k8s.io/client-go/kubernetes"
	kube_flag "k8s.io/component-base/cli/flag"
	"k8s.io/klog"
)

var (
	updaterInterval = flag.Duration("updater-interval", 1*time.Minute,
		`How often updater should run`)

	minReplicas = flag.Int("min-replicas", 2,
		`Minimum number of replicas to perform update`)

	evictionToleranceFraction = flag.Float64("eviction-tolerance", 0.5,
		`Fraction of replica count that can be evicted for update, if more than one pod can be evicted.`)

	evictionRateLimit = flag.Float64("eviction-rate-limit", -1,
		`Number of pods that can be evicted per seconds. A rate limit set to 0 or -1 will disable
		the rate limiter.`)

	evictionRateBurst = flag.Int("eviction-rate-burst", 1, `Burst of pods that can be evicted.`)

	address      = flag.String("address", ":8943", "The address to expose Prometheus metrics.")
	kubeconfig   = flag.String("kubeconfig", "", "Path to a kubeconfig. Only required if out-of-cluster.")
	kubeApiQps   = flag.Float64("kube-api-qps", 5.0, `QPS limit when making requests to Kubernetes apiserver`)
	kubeApiBurst = flag.Float64("kube-api-burst", 10.0, `QPS burst limit when making requests to Kubernetes apiserver`)

	useAdmissionControllerStatus = flag.Bool("use-admission-controller-status", true,
		"If true, updater will only evict pods when admission controller status is valid.")

	namespace          = os.Getenv("NAMESPACE")
	vpaObjectNamespace = flag.String("vpa-object-namespace", apiv1.NamespaceAll, "Namespace to search for VPA objects. Empty means all namespaces will be used.")
)

const defaultResyncPeriod time.Duration = 10 * time.Minute

func main() {
	klog.InitFlags(nil)
	kube_flag.InitFlags()
	klog.V(1).Infof("Vertical Pod Autoscaler %s Updater", common.VerticalPodAutoscalerVersion)

	healthCheck := metrics.NewHealthCheck(*updaterInterval*5, true)
	metrics.Initialize(*address, healthCheck)
	metrics_updater.Register()

	config := common.CreateKubeConfigOrDie(*kubeconfig, float32(*kubeApiQps), int(*kubeApiBurst))
	kubeClient := kube_client.NewForConfigOrDie(config)
	vpaClient := vpa_clientset.NewForConfigOrDie(config)
	factory := informers.NewSharedInformerFactory(kubeClient, defaultResyncPeriod)
	targetSelectorFetcher := target.NewVpaTargetSelectorFetcher(config, kubeClient, factory)
	var limitRangeCalculator limitrange.LimitRangeCalculator
	limitRangeCalculator, err := limitrange.NewLimitsRangeCalculator(factory)
	if err != nil {
		klog.Errorf("Failed to create limitRangeCalculator, falling back to not checking limits. Error message: %s", err)
		limitRangeCalculator = limitrange.NewNoopLimitsCalculator()
	}
	admissionControllerStatusNamespace := status.AdmissionControllerStatusNamespace
	if namespace != "" {
		admissionControllerStatusNamespace = namespace
	}
	// TODO: use SharedInformerFactory in updater
	updater, err := updater.NewUpdater(
		kubeClient,
		vpaClient,
		*minReplicas,
		*evictionRateLimit,
		*evictionRateBurst,
		*evictionToleranceFraction,
		*useAdmissionControllerStatus,
		admissionControllerStatusNamespace,
		vpa_api_util.NewCappingRecommendationProcessor(limitRangeCalculator),
		nil,
		targetSelectorFetcher,
		priority.NewProcessor(),
		*vpaObjectNamespace,
	)
	if err != nil {
		klog.Fatalf("Failed to create updater: %v", err)
	}
	ticker := time.Tick(*updaterInterval)
	for range ticker {
		ctx, cancel := context.WithTimeout(context.Background(), *updaterInterval)
		defer cancel()
		updater.RunOnce(ctx)
		healthCheck.UpdateLastActivity()
	}
}
